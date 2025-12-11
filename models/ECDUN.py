# -*- coding: utf-8 -*-
"""
ECDUN: Edge-Constrained Deblurring U-Net with OCR Guidance
Enhanced version with OCR character recognition for license plate deblurring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os
from .DM import ImprovedEncoder, ImprovedDecoder, ImprovedBottleneck
from .Up_sample import ConvUpInterp, ConvUp, ConvDown
from .residualprojectionModule import UCNet
from .edgemap import EdgeMap
from .edgefeatureextractionModule import EAFM
from .intermediatevariableupdateModule import EGIM
from .variableguidereconstructionModule import IGRM
from .DDM import LicensePlateDeblurModule
from .EPDD import EdgePreservingDynamicDeblur
from .ocr_module import OCRGuidedRefinement, OCRMetrics


def make_model(args, parent=False):
    return ECDUN(args)


class ECDUN(nn.Module):
    """
    ECDUN model with OCR-guided refinement for license plate deblurring
    
    Key enhancements:
    1. OCR character recognition to guide deblurring
    2. Confidence-based refinement for low-quality regions
    3. OCR-aware loss function for better training
    """
    _global_save_intermediate = False
    _global_vis_save_dir = './visualization'
    _global_current_filename = None
    
    def __init__(self, args):
        super(ECDUN, self).__init__()

        self.args = args
        self.rgb_range = float(getattr(args, 'rgb_range', 255.0))
        self.channel0 = args.n_colors
        self.up_factor = args.scale[0]
        self.down_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size / args.n_GPUs)
        
        # OCR configuration
        self.use_ocr = getattr(args, 'use_ocr', True)
        self.ocr_backend = getattr(args, 'ocr_backend', 'easyocr')  # 'easyocr' or 'paddleocr'
        self.ocr_refine_iterations = getattr(args, 'ocr_refine_iterations', [2, 3])  # Refine at these iterations
        self.ocr_weight = getattr(args, 'ocr_weight', 0.1)
        
        self.save_intermediate = False
        self.vis_save_dir = './visualization'
        self.current_filename = None

        self.dynamic_deblur = LicensePlateDeblurModule(in_ch=64, kernel_size=5)

        self.act = nn.ReLU()
        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3
        T = 4
        
        self.Encoder0 = ImprovedEncoder(in_ch=G0, embed_dim=G0, num_blocks=2)
        self.Encoder1 = ImprovedEncoder(in_ch=G0, embed_dim=G0, num_blocks=3)
        self.Encoder2 = ImprovedEncoder(in_ch=G0, embed_dim=G0, num_blocks=3)
        
        self.bottleneck = nn.Sequential(
            ImprovedBottleneck(G0, num_blocks=4),
        )
        
        self.Decoder2 = ImprovedDecoder(in_ch=G0, skip_ch=G0, out_ch=G0)
        self.Decoder1 = ImprovedDecoder(in_ch=G0, skip_ch=G0, out_ch=G0)
        self.Decoder0 = ImprovedDecoder(in_ch=G0, skip_ch=G0, out_ch=G0)
        
        # Multi-scale fusion with InstanceNorm
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(G0 * 4, G0 * 2, 1, bias=False),
            nn.InstanceNorm2d(G0 * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(G0 * 2, G0, 3, padding=1, bias=False),
            nn.InstanceNorm2d(G0, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Enhanced reconstruction module
        self.reconstruct = nn.Sequential(
            nn.Conv2d(G0, G0, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(G0, G0, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(G0, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.Fe_e = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
            ) for _ in range(T)]
        )

        self.RNNF = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
                self.act,
                nn.Conv2d(G0, 3, 3, padding=1),
                nn.Tanh()
            ) for i in range(T)]
        )

        self.Fe_f = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1)
            ) for i in range(T - 1)]
        )

        # Learnable parameters
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(0.05)) for _ in range(T)])
        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.05)) for _ in range(T)])
        self.delta_1 = nn.ParameterList([nn.Parameter(torch.tensor(0.3)) for _ in range(T)])
        self.delta_2 = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(T)])
        self.delta_3 = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(T)])
        self.gama = nn.Parameter(torch.tensor(0.01))
        self.res_scale = nn.Parameter(torch.tensor(0.3))
        
        if self.up_factor == 1:
            self.conv_up = nn.Identity()
            self.conv_down = nn.Identity()
        else:
            self.conv_up = ConvUp(3, self.up_factor)
            self.conv_down = ConvDown(3, self.up_factor)

        self.blur = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        with torch.no_grad():
            gaussian = torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ]) / 16.0
            
            for c in range(3):
                self.blur.weight[c, 0] = gaussian
        
        self.blur.weight.requires_grad = False
        
        # Pixel shuffle upsampling
        r = int(self.up_factor)
        if r > 1:
            self.pixel_shuffle_up = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3 * (r**2), 3, padding=1),
                nn.PixelShuffle(r)
            )
        else:
            self.pixel_shuffle_up = nn.Identity()
            
        self.UCNet = UCNet(3, 64)
        self.delta_down = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.edgemap = EdgeMap()
        self.EAFM = EAFM()
        self.EGIM = EGIM()
        self.IGRM = IGRM()

        self.delta_up_scale = 0.3
        
        # ========== OCR Module Integration ==========
        if self.use_ocr:
            self.ocr_refiner = OCRGuidedRefinement(
                ocr_backend=self.ocr_backend,
                use_confidence_map=True,
                confidence_threshold=0.5,
                refine_features=64
            )
            print(f"[ECDUN] OCR-guided refinement enabled with backend: {self.ocr_backend}")
        else:
            self.ocr_refiner = None
            print("[ECDUN] OCR-guided refinement disabled")

        # Weight initialization
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        self.apply(_init_weights)
        
        # Ensure blur kernel is initialized
        with torch.no_grad():
            gaussian = torch.tensor([
                [1., 2., 1.],
                [2., 4., 2.],
                [1., 2., 1.]
            ]) / 16.0
            
            for c in range(3):
                self.blur.weight[c, 0] = gaussian
        self.blur.weight.requires_grad = False
        
    def _visualize_image(self, image, name, iteration, batch_idx=0):
        """Visualize image or residual"""
        if not ECDUN._global_save_intermediate:
            return
        
        try:
            os.makedirs(ECDUN._global_vis_save_dir, exist_ok=True)
            
            img = image[batch_idx].detach().cpu()
            
            # Check if this is a residual or normal image
            is_residual = "residual" in name.lower() or "rpm_residual" in name.lower()
            
            if is_residual:
                img_min = float(img.min())
                img_max = float(img.max())
                
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min + 1e-8)
                else:
                    img = torch.zeros_like(img) + 0.5
            else:
                img = torch.clamp(img, 0, 1)
            
            prefix = ECDUN._global_current_filename if ECDUN._global_current_filename else 'unknown'
            save_path = os.path.join(ECDUN._global_vis_save_dir, f'{prefix}_iter{iteration}_{name}.png')
            vutils.save_image(img, save_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to visualize {name}: {type(e).__name__}")
    
    def _visualize_feature(self, feature, name, iteration, batch_idx=0):
        """Visualize feature maps"""
        if not ECDUN._global_save_intermediate:
            return
        
        try:
            os.makedirs(ECDUN._global_vis_save_dir, exist_ok=True)
            
            feat = feature[batch_idx].detach().cpu()
            
            if feat.shape[0] > 3:
                feat = feat[:3]
            elif feat.shape[0] == 1:
                feat = feat.repeat(3, 1, 1)
            
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            
            prefix = ECDUN._global_current_filename if ECDUN._global_current_filename else 'unknown'
            save_path = os.path.join(ECDUN._global_vis_save_dir, f'{prefix}_iter{iteration}_{name}.png')
            
            vutils.save_image(feat, save_path)
                
        except Exception as e:
            print(f"[ERROR] _visualize_feature: {e}")

    def forward(self, y, idx_scale=0):
        """
        Forward pass with OCR-guided refinement
        
        Args:
            y: input blurred image [B, C, H, W] in range [0, rgb_range]
            idx_scale: scale index (not used)
            
        Returns:
            x_out: deblurred output [B, C, H, W] in range [0, rgb_range]
        """
        # Input protection
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("[ERROR] Input contains NaN/Inf!")
            y = torch.nan_to_num(y, nan=0.0, posinf=self.rgb_range, neginf=0.0)
        
        # Normalize to [0, 1]
        y_norm = y / self.rgb_range
        y_norm = torch.clamp(y_norm, 0, 1)
        
        fea_list = []
        V_list = []
        ocr_info_list = []
        
        x_init = y_norm.clone()
        f_init = self.edgemap(x_init)
        v_init = x_init.clone()
        target_size = y_norm.shape[2:]
        
        # ==================== Iterative Optimization ====================
        for i in range(len(self.Fe_e)):
            # Feature extraction
            fea = self.Fe_e[i](x_init)
            fea_list.append(fea)
            
            if i != 0:
                fea = self.Fe_f[i - 1](torch.cat(fea_list, 1))
            
            # Encoder
            enc0, d0 = self.Encoder0(fea)
            enc1, d1 = self.Encoder1(d0)
            enc2, d2 = self.Encoder2(d1)
            
            # Bottleneck + dynamic deblurring
            bottleneck_feat = self.bottleneck(d2)
            media_end = self.dynamic_deblur(bottleneck_feat)
            
            # Decoder
            dec2 = self.Decoder2(media_end, enc2)
            dec1 = self.Decoder1(dec2, enc1)
            dec0 = self.Decoder0(dec1, enc0)
            
            V_list.append(dec0)
            
            # Multi-scale fusion
            if i == 0:
                enc1_up = F.interpolate(enc1, size=dec0.shape[2:], mode='bilinear', align_corners=False)
                enc2_up = F.interpolate(enc2, size=dec0.shape[2:], mode='bilinear', align_corners=False)
                x_cat = torch.cat([dec0, enc0, enc1_up, enc2_up], dim=1)
                fused_feat = self.multi_scale_fusion(x_cat)
                decode_img = self.reconstruct(fused_feat)
            else:
                decode_img = self.RNNF[i - 1](torch.cat(V_list, 1))
            
            # Update x_init with residual
            residual = torch.clamp(decode_img, -0.5, 0.5)
            x_init = torch.clamp(x_init + self.res_scale * residual, 0, 1)
            
            self._visualize_image(x_init, f'after_denoise_lr_{i+1}', i+1)
            
            # Edge guidance
            z = f_init - self.delta_1[i] * (f_init - self.edgemap(x_init))
            f_init = self.EAFM(z)
            
            t = v_init - self.delta_2[i] * (v_init - x_init)
            v_init = torch.clamp(self.EGIM(t, f_init), 0, 1)
            
            # Intermediate reconstruction
            if self.up_factor > 1:
                temp_size = self.conv_down(x_init).shape[2:]
                temp_y = F.interpolate(y_norm, size=temp_size, mode='bilinear', align_corners=False)
                up_result = self.conv_up(self.conv_down(x_init) - temp_y)
            else:
                up_result = x_init - y_norm
            
            if up_result.shape[2:] != x_init.shape[2:]:
                up_result = F.interpolate(up_result, size=x_init.shape[2:], mode='bilinear', align_corners=False)
            
            r = x_init - self.delta_3[i] * (up_result + self.mu[i] * (v_init - x_init))
            r = torch.clamp(r, 0, 1)
            
            x_init = torch.clamp(self.IGRM(r, v_init), 0, 1)
            self._visualize_image(x_init, f'after_igrm_lr_{i+1}', i+1)
            
            # Residual projection module
            blurred_x = self.blur(x_init)
            
            if self.up_factor > 1:
                down_out = F.interpolate(blurred_x, size=target_size, mode='bilinear', align_corners=False)
                difference = y_norm - down_out
                delta_uc = F.interpolate(difference, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
            else:
                difference = y_norm - blurred_x
                delta_uc = difference
            
            delta_down = self.delta_down(delta_uc)
            delta = self.UCNet(delta_down)
            
            if delta.shape[2:] != x_init.shape[2:]:
                delta = F.interpolate(delta, size=x_init.shape[2:], mode='bilinear', align_corners=False)
            
            x_init = torch.clamp(x_init + self.delta_up_scale * delta, 0, 1)
            self._visualize_image(x_init, f'after_rpm_lr_{i+1}', i+1)
            
            # ========== OCR-Guided Refinement ==========
            if self.use_ocr and self.ocr_refiner is not None and i in self.ocr_refine_iterations:
                x_init_refined, ocr_info = self.ocr_refiner(
                    x_init, 
                    use_ocr=True, 
                    training=self.training
                )
                x_init = x_init_refined
                ocr_info_list.append(ocr_info)
                
                # Visualize OCR refinement
                self._visualize_image(x_init, f'after_ocr_refine_{i+1}', i+1)
                
                if ECDUN._global_save_intermediate and ocr_info.get('confidence_map') is not None:
                    self._visualize_feature(
                        ocr_info['confidence_map'], 
                        f'ocr_confidence_{i+1}', 
                        i+1
                    )
                
                # Print OCR results
                if not self.training:
                    for j, result in enumerate(ocr_info['results']):
                        if result['text']:
                            print(f"[OCR] Iter {i+1}, Batch {j}: '{result['text']}' "
                                  f"(conf: {result['confidence']:.3f})")
            
            # Numerical stability check
            if torch.isnan(x_init).any() or torch.isinf(x_init).any():
                print(f"[WARNING] NaN/Inf at iteration {i}, using safe fallback")
                x_init = torch.clamp(y_norm, 0, 1)
                break
        
        # ==================== Final Output ====================
        if self.up_factor > 1:
            # Super-resolution task
            x_hr = self.pixel_shuffle_up(x_init)
            x_hr = torch.clamp(x_hr, 0, 1)
            
            # Final OCR refinement at high resolution
            if self.use_ocr and self.ocr_refiner is not None:
                x_hr, final_ocr_info = self.ocr_refiner(x_hr, use_ocr=True, training=self.training)
                ocr_info_list.append(final_ocr_info)
                
                if not self.training:
                    for j, result in enumerate(final_ocr_info['results']):
                        if result['text']:
                            print(f"[OCR] Final HR: '{result['text']}' "
                                  f"(conf: {result['confidence']:.3f})")
            
            x_out = x_hr * self.rgb_range
        else:
            # Deblurring task (LR -> LR)
            x_out = x_init * self.rgb_range
        
        x_out = torch.clamp(x_out, 0.0, self.rgb_range)
        
        self._visualize_image(x_out / self.rgb_range, 'final_output', len(self.Fe_e))
        
        # Store OCR info for potential loss computation
        self._last_ocr_info = ocr_info_list
        
        return x_out
    
    def compute_loss_with_ocr(self, pred, target):
        """
        Compute loss with OCR guidance
        
        Args:
            pred: predicted output [B, C, H, W]
            target: ground truth [B, C, H, W]
            
        Returns:
            total_loss: combined loss
            loss_dict: dictionary with loss components
        """
        # Normalize to [0, 1]
        pred_norm = pred / self.rgb_range
        target_norm = target / self.rgb_range
        
        # Standard reconstruction loss
        recon_loss = F.l1_loss(pred_norm, target_norm)
        
        # OCR-guided loss
        if self.use_ocr and self.ocr_refiner is not None:
            ocr_loss, ocr_loss_dict = self.ocr_refiner.compute_ocr_loss(
                pred_norm, target_norm, ocr_weight=self.ocr_weight
            )
            total_loss = recon_loss + self.ocr_weight * ocr_loss
            
            loss_dict = {
                'total_loss': float(total_loss),
                'recon_loss': float(recon_loss),
                **ocr_loss_dict
            }
        else:
            total_loss = recon_loss
            loss_dict = {
                'total_loss': float(total_loss),
                'recon_loss': float(recon_loss)
            }
        
        return total_loss, loss_dict
