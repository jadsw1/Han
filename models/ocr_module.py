# -*- coding: utf-8 -*-
"""
OCR-Guided Refinement Module for License Plate Deblurring

This module integrates OCR (Optical Character Recognition) to guide and improve
the deblurring process by:
1. Detecting characters in the deblurred image
2. Computing OCR confidence scores
3. Refining the image based on low-confidence regions
4. Providing OCR-guided loss for training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OCRGuidedRefinement(nn.Module):
    """
    OCR-guided refinement module that uses character recognition 
    to improve license plate deblurring quality.
    """
    def __init__(self, ocr_backend='easyocr', use_confidence_map=True, 
                 confidence_threshold=0.5, refine_features=64):
        """
        Args:
            ocr_backend: 'easyocr' or 'paddleocr' for character recognition
            use_confidence_map: whether to use spatial confidence maps
            confidence_threshold: minimum confidence for character detection
            refine_features: number of features for refinement network
        """
        super(OCRGuidedRefinement, self).__init__()
        
        self.ocr_backend = ocr_backend
        self.use_confidence_map = use_confidence_map
        self.confidence_threshold = confidence_threshold
        
        # Initialize OCR reader (lazy loading to avoid loading weights at init)
        self.ocr_reader = None
        self._ocr_initialized = False
        
        # Confidence-aware refinement network
        self.confidence_encoder = nn.Sequential(
            nn.Conv2d(4, refine_features, 3, padding=1),  # RGB + confidence map
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(refine_features, refine_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.attention_module = nn.Sequential(
            nn.Conv2d(refine_features, refine_features // 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(refine_features // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.refinement_decoder = nn.Sequential(
            nn.Conv2d(refine_features, refine_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(refine_features, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Character-level feature extractor for training
        self.char_feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
    def _init_ocr_reader(self):
        """Lazy initialization of OCR reader"""
        if self._ocr_initialized:
            return
        
        try:
            if self.ocr_backend == 'easyocr':
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            elif self.ocr_backend == 'paddleocr':
                from paddleocr import PaddleOCR
                self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', 
                                           use_gpu=torch.cuda.is_available())
            else:
                raise ValueError(f"Unsupported OCR backend: {self.ocr_backend}")
            
            self._ocr_initialized = True
            print(f"[OCR] Initialized {self.ocr_backend} backend successfully")
        except Exception as e:
            print(f"[WARNING] Failed to initialize OCR reader: {e}")
            print("[WARNING] OCR-guided refinement will be disabled")
            self._ocr_initialized = False
    
    def detect_characters(self, image_tensor, return_confidence_map=True):
        """
        Detect characters in the image and return confidence scores
        
        Args:
            image_tensor: [B, 3, H, W] tensor with values in [0, 1]
            return_confidence_map: whether to return spatial confidence map
            
        Returns:
            results: list of detection results for each image in batch
            confidence_maps: [B, 1, H, W] spatial confidence maps (if enabled)
        """
        if not self._ocr_initialized:
            self._init_ocr_reader()
        
        if self.ocr_reader is None:
            # OCR not available, return dummy results
            B, _, H, W = image_tensor.shape
            dummy_map = torch.ones(B, 1, H, W, device=image_tensor.device) * 0.5
            return [{'text': '', 'confidence': 0.5}] * B, dummy_map
        
        batch_size = image_tensor.shape[0]
        results = []
        confidence_maps = []
        
        for i in range(batch_size):
            # Convert tensor to numpy image
            img = image_tensor[i].detach().cpu()
            img = (img * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # Run OCR
            try:
                if self.ocr_backend == 'easyocr':
                    detections = self.ocr_reader.readtext(img)
                    batch_result = {
                        'text': ' '.join([det[1] for det in detections]),
                        'confidence': np.mean([det[2] for det in detections]) if detections else 0.0,
                        'boxes': [det[0] for det in detections],
                        'scores': [det[2] for det in detections]
                    }
                elif self.ocr_backend == 'paddleocr':
                    detections = self.ocr_reader.ocr(img, cls=True)
                    if detections and detections[0]:
                        batch_result = {
                            'text': ' '.join([det[1][0] for det in detections[0]]),
                            'confidence': np.mean([det[1][1] for det in detections[0]]),
                            'boxes': [det[0] for det in detections[0]],
                            'scores': [det[1][1] for det in detections[0]]
                        }
                    else:
                        batch_result = {'text': '', 'confidence': 0.0, 'boxes': [], 'scores': []}
                
                results.append(batch_result)
                
                # Create confidence map
                if return_confidence_map:
                    conf_map = self._create_confidence_map(
                        batch_result, image_tensor.shape[2:], image_tensor.device
                    )
                    confidence_maps.append(conf_map)
                    
            except Exception as e:
                print(f"[WARNING] OCR detection failed for image {i}: {e}")
                results.append({'text': '', 'confidence': 0.0, 'boxes': [], 'scores': []})
                if return_confidence_map:
                    H, W = image_tensor.shape[2:]
                    conf_map = torch.ones(1, H, W, device=image_tensor.device) * 0.5
                    confidence_maps.append(conf_map)
        
        if return_confidence_map:
            confidence_maps = torch.stack(confidence_maps, dim=0)
        else:
            confidence_maps = None
            
        return results, confidence_maps
    
    def _create_confidence_map(self, ocr_result, img_size, device):
        """
        Create a spatial confidence map based on OCR detections
        
        Args:
            ocr_result: dictionary with 'boxes' and 'scores'
            img_size: (H, W) tuple
            device: torch device
            
        Returns:
            confidence_map: [1, H, W] tensor
        """
        H, W = img_size
        conf_map = torch.ones(1, H, W, device=device) * 0.3  # Low confidence baseline
        
        boxes = ocr_result.get('boxes', [])
        scores = ocr_result.get('scores', [])
        
        for box, score in zip(boxes, scores):
            # Convert box coordinates to integer pixel coordinates
            box = np.array(box, dtype=np.int32)
            x_min = max(0, min(box[:, 0]))
            x_max = min(W, max(box[:, 0]))
            y_min = max(0, min(box[:, 1]))
            y_max = min(H, max(box[:, 1]))
            
            # Fill the box region with confidence score
            conf_map[0, y_min:y_max, x_min:x_max] = score
        
        return conf_map
    
    def forward(self, image, use_ocr=True, training=False):
        """
        Refine image using OCR guidance
        
        Args:
            image: [B, 3, H, W] input image tensor in [0, 1]
            use_ocr: whether to use OCR for guidance (set False for speed)
            training: whether in training mode
            
        Returns:
            refined_image: [B, 3, H, W] refined image
            ocr_info: dictionary with OCR results and confidence
        """
        B, C, H, W = image.shape
        
        # Get OCR confidence map if enabled
        if use_ocr and self.use_confidence_map:
            ocr_results, confidence_map = self.detect_characters(image, return_confidence_map=True)
        else:
            ocr_results = [{'text': '', 'confidence': 1.0}] * B
            confidence_map = torch.ones(B, 1, H, W, device=image.device)
        
        # Concatenate image with confidence map
        input_with_conf = torch.cat([image, confidence_map], dim=1)
        
        # Extract features
        features = self.confidence_encoder(input_with_conf)
        
        # Generate attention map (focus on low-confidence regions)
        attention = self.attention_module(features)
        attention_inv = 1.0 - attention  # Focus on low confidence areas
        
        # Apply attention to features
        attended_features = features * attention_inv
        
        # Generate refinement residual
        residual = self.refinement_decoder(attended_features)
        residual = residual * 0.1  # Small scale refinement
        
        # Apply refinement
        refined_image = image + residual
        refined_image = torch.clamp(refined_image, 0, 1)
        
        # Compute average OCR confidence
        avg_confidence = np.mean([r['confidence'] for r in ocr_results])
        
        ocr_info = {
            'results': ocr_results,
            'confidence_map': confidence_map,
            'attention_map': attention,
            'avg_confidence': avg_confidence,
            'residual_norm': float(residual.abs().mean())
        }
        
        return refined_image, ocr_info
    
    def compute_ocr_loss(self, pred_image, target_image, ocr_weight=0.1):
        """
        Compute OCR-guided loss to encourage better character recognition
        
        Args:
            pred_image: [B, 3, H, W] predicted image
            target_image: [B, 3, H, W] target ground truth image
            ocr_weight: weight for OCR loss component
            
        Returns:
            loss: scalar loss value
            loss_dict: dictionary with loss components
        """
        # Standard reconstruction loss
        recon_loss = F.l1_loss(pred_image, target_image)
        
        # Get OCR confidence for predicted and target images
        pred_results, pred_conf_map = self.detect_characters(pred_image)
        target_results, target_conf_map = self.detect_characters(target_image)
        
        # OCR confidence loss (encourage high confidence)
        pred_avg_conf = np.mean([r['confidence'] for r in pred_results])
        target_avg_conf = np.mean([r['confidence'] for r in target_results])
        
        conf_loss = max(0, target_avg_conf - pred_avg_conf)
        conf_loss = torch.tensor(conf_loss, device=pred_image.device)
        
        # Spatial confidence map loss
        conf_map_loss = F.mse_loss(pred_conf_map, target_conf_map)
        
        # Total OCR loss
        ocr_loss = conf_loss + conf_map_loss
        
        # Combined loss
        total_loss = recon_loss + ocr_weight * ocr_loss
        
        loss_dict = {
            'total_loss': float(total_loss),
            'recon_loss': float(recon_loss),
            'ocr_loss': float(ocr_loss),
            'conf_loss': float(conf_loss),
            'conf_map_loss': float(conf_map_loss),
            'pred_confidence': pred_avg_conf,
            'target_confidence': target_avg_conf
        }
        
        return total_loss, loss_dict


class OCRMetrics:
    """Utility class to compute OCR-based metrics"""
    
    @staticmethod
    def compute_character_accuracy(pred_text, target_text):
        """Compute character-level accuracy"""
        if not target_text:
            return 0.0 if pred_text else 1.0
        
        # Remove spaces and convert to uppercase
        pred_text = pred_text.replace(' ', '').upper()
        target_text = target_text.replace(' ', '').upper()
        
        # Compute character accuracy
        correct = sum(p == t for p, t in zip(pred_text, target_text))
        total = max(len(pred_text), len(target_text))
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def compute_edit_distance(pred_text, target_text):
        """Compute Levenshtein edit distance"""
        pred_text = pred_text.replace(' ', '').upper()
        target_text = target_text.replace(' ', '').upper()
        
        m, n = len(pred_text), len(target_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_text[i-1] == target_text[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
