# -*- coding: utf-8 -*-
"""
Example usage of ECDUN model with OCR integration for license plate deblurring
"""
import torch
import argparse
from models.ECDUN import ECDUN


def create_args():
    """Create argument object with default parameters"""
    args = argparse.Namespace()
    
    # Basic parameters
    args.n_colors = 3
    args.scale = [1]  # [1] for deblurring, [2] or [4] for super-resolution
    args.patch_size = 128
    args.batch_size = 4
    args.n_GPUs = 1
    args.rgb_range = 255.0
    
    # OCR parameters (NEW)
    args.use_ocr = True  # Enable OCR-guided refinement
    args.ocr_backend = 'easyocr'  # Options: 'easyocr' or 'paddleocr'
    args.ocr_refine_iterations = [2, 3]  # Apply OCR refinement at these iterations
    args.ocr_weight = 0.1  # Weight for OCR loss during training
    
    return args


def example_inference():
    """Example: Inference on a single blurred license plate image"""
    print("=" * 60)
    print("Example 1: Inference with OCR-guided refinement")
    print("=" * 60)
    
    # Create model
    args = create_args()
    args.use_ocr = True
    model = ECDUN(args)
    model.eval()
    
    # Create dummy input (replace with real image)
    # Input should be [B, C, H, W] with values in [0, 255]
    dummy_input = torch.rand(1, 3, 128, 256) * 255.0
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\n✓ Inference completed successfully!")
    print("  OCR refinement was applied at iterations 2 and 3")
    print("  Check console for detected characters and confidence scores")


def example_inference_without_ocr():
    """Example: Inference without OCR (faster but less accurate)"""
    print("\n" + "=" * 60)
    print("Example 2: Inference WITHOUT OCR (faster)")
    print("=" * 60)
    
    # Create model without OCR
    args = create_args()
    args.use_ocr = False  # Disable OCR
    model = ECDUN(args)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.rand(1, 3, 128, 256) * 255.0
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    print("\n✓ Inference completed (OCR disabled)")


def example_training():
    """Example: Training with OCR-guided loss"""
    print("\n" + "=" * 60)
    print("Example 3: Training with OCR-guided loss")
    print("=" * 60)
    
    # Create model
    args = create_args()
    args.use_ocr = True
    model = ECDUN(args)
    model.train()
    
    # Create dummy batch
    blurred = torch.rand(2, 3, 128, 256) * 255.0
    sharp = torch.rand(2, 3, 128, 256) * 255.0
    
    # Forward pass
    output = model(blurred)
    
    # Compute loss with OCR guidance
    loss, loss_dict = model.compute_loss_with_ocr(output, sharp)
    
    print(f"Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    # Backward pass (example)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    print("\n✓ Training step completed!")
    print("  OCR-guided loss helps improve character recognition accuracy")


def example_ocr_backends():
    """Example: Comparing different OCR backends"""
    print("\n" + "=" * 60)
    print("Example 4: OCR Backend Comparison")
    print("=" * 60)
    
    # Test EasyOCR
    print("\n1. Using EasyOCR backend:")
    args = create_args()
    args.use_ocr = True
    args.ocr_backend = 'easyocr'
    model_easy = ECDUN(args)
    print("   ✓ EasyOCR model created")
    
    # Test PaddleOCR
    print("\n2. Using PaddleOCR backend:")
    args.ocr_backend = 'paddleocr'
    model_paddle = ECDUN(args)
    print("   ✓ PaddleOCR model created")
    
    print("\nOCR Backend Selection:")
    print("  - EasyOCR: Better for English, easier installation")
    print("  - PaddleOCR: Better for Chinese/multilingual, faster")


def example_super_resolution_with_ocr():
    """Example: Super-resolution with OCR guidance"""
    print("\n" + "=" * 60)
    print("Example 5: Super-resolution + OCR")
    print("=" * 60)
    
    # Create SR model
    args = create_args()
    args.scale = [2]  # 2x super-resolution
    args.use_ocr = True
    model = ECDUN(args)
    model.eval()
    
    # Low-resolution input
    lr_input = torch.rand(1, 3, 64, 128) * 255.0
    
    # Run SR + deblurring
    with torch.no_grad():
        hr_output = model(lr_input)
        print(f"Input (LR) shape: {lr_input.shape}")
        print(f"Output (HR) shape: {hr_output.shape}")
        print(f"Scale factor: {hr_output.shape[2] / lr_input.shape[2]:.1f}x")
    
    print("\n✓ Super-resolution completed!")
    print("  OCR refinement is applied at both LR and HR stages")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ECDUN Model with OCR Integration - Examples")
    print("=" * 60)
    print("\nThis script demonstrates how to use the ECDUN model")
    print("with OCR-guided refinement for license plate deblurring.\n")
    
    # Run examples
    example_inference()
    example_inference_without_ocr()
    example_training()
    example_ocr_backends()
    example_super_resolution_with_ocr()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ OCR character recognition for guidance")
    print("  ✓ Confidence-based refinement")
    print("  ✓ OCR-aware loss function")
    print("  ✓ Support for EasyOCR and PaddleOCR")
    print("  ✓ Works for both deblurring and super-resolution")
    print("\nNext Steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Prepare your license plate dataset")
    print("  3. Configure OCR backend (EasyOCR or PaddleOCR)")
    print("  4. Train the model with OCR guidance")
    print("  5. Evaluate character recognition accuracy")
