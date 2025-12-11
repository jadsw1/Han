# -*- coding: utf-8 -*-
"""
Test script to validate ECDUN model with OCR integration
Tests model creation, forward pass, and OCR functionality
"""
import torch
import argparse
import sys


def create_test_args(use_ocr=True, scale=1):
    """Create test arguments"""
    args = argparse.Namespace()
    args.n_colors = 3
    args.scale = [scale]
    args.patch_size = 128
    args.batch_size = 2
    args.n_GPUs = 1
    args.rgb_range = 255.0
    args.use_ocr = use_ocr
    args.ocr_backend = 'easyocr'
    args.ocr_refine_iterations = [2, 3]
    args.ocr_weight = 0.1
    return args


def test_model_creation():
    """Test 1: Model creation"""
    print("\n" + "=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)
    
    try:
        from models.ECDUN import ECDUN
        args = create_test_args(use_ocr=False)  # Start without OCR
        model = ECDUN(args)
        print("✓ Model created successfully")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_deblur():
    """Test 2: Forward pass for deblurring (scale=1)"""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass - Deblurring (1x)")
    print("=" * 60)
    
    try:
        from models.ECDUN import ECDUN
        args = create_test_args(use_ocr=False, scale=1)
        model = ECDUN(args)
        model.eval()
        
        # Test input
        batch_size = 2
        h, w = 64, 128
        input_tensor = torch.rand(batch_size, 3, h, w) * 255.0
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Input range: [{input_tensor.min():.2f}, {input_tensor.max():.2f}]")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
        
        # Validate
        assert output.shape == input_tensor.shape, "Output shape mismatch!"
        assert output.min() >= 0 and output.max() <= 255, "Output out of range!"
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        
        print("✓ Deblurring forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_sr():
    """Test 3: Forward pass for super-resolution (scale=2)"""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass - Super-Resolution (2x)")
    print("=" * 60)
    
    try:
        from models.ECDUN import ECDUN
        args = create_test_args(use_ocr=False, scale=2)
        model = ECDUN(args)
        model.eval()
        
        # Test input (low resolution)
        batch_size = 2
        h, w = 64, 128
        input_tensor = torch.rand(batch_size, 3, h, w) * 255.0
        
        print(f"Input shape: {input_tensor.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output shape: {output.shape}")
        expected_shape = (batch_size, 3, h * 2, w * 2)
        print(f"Expected shape: {expected_shape}")
        
        # Validate
        assert output.shape == expected_shape, f"Output shape mismatch! Got {output.shape}, expected {expected_shape}"
        assert output.min() >= 0 and output.max() <= 255, "Output out of range!"
        
        print("✓ Super-resolution forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_module():
    """Test 4: OCR module"""
    print("\n" + "=" * 60)
    print("TEST 4: OCR Module")
    print("=" * 60)
    
    try:
        from models.ocr_module import OCRGuidedRefinement
        
        # Create OCR module (without actual OCR backend to avoid installation issues)
        ocr_module = OCRGuidedRefinement(ocr_backend='easyocr', use_confidence_map=True)
        
        # Test input
        batch_size = 2
        h, w = 64, 128
        input_tensor = torch.rand(batch_size, 3, h, w)  # [0, 1] range
        
        print(f"Input shape: {input_tensor.shape}")
        
        # Forward pass (without actual OCR)
        with torch.no_grad():
            refined, ocr_info = ocr_module(input_tensor, use_ocr=False, training=False)
        
        print(f"Refined shape: {refined.shape}")
        print(f"OCR info keys: {list(ocr_info.keys())}")
        
        # Validate
        assert refined.shape == input_tensor.shape, "Refined shape mismatch!"
        assert 'confidence_map' in ocr_info, "Missing confidence map!"
        assert 'attention_map' in ocr_info, "Missing attention map!"
        
        print("✓ OCR module test successful (without actual OCR backend)")
        print("  Note: Install EasyOCR or PaddleOCR for full OCR functionality")
        return True
    except Exception as e:
        print(f"✗ OCR module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_ocr():
    """Test 5: Model with OCR enabled"""
    print("\n" + "=" * 60)
    print("TEST 5: Model with OCR Integration")
    print("=" * 60)
    
    try:
        from models.ECDUN import ECDUN
        args = create_test_args(use_ocr=True, scale=1)
        model = ECDUN(args)
        model.eval()
        
        # Test input
        batch_size = 1
        h, w = 64, 128
        input_tensor = torch.rand(batch_size, 3, h, w) * 255.0
        
        print(f"Input shape: {input_tensor.shape}")
        print("OCR backend: easyocr (lazy loading)")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output shape: {output.shape}")
        
        # Validate
        assert output.shape == input_tensor.shape, "Output shape mismatch!"
        
        print("✓ Model with OCR integration successful")
        print("  Note: OCR detection runs if backend is installed")
        return True
    except Exception as e:
        print(f"✗ Model with OCR failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test 6: Loss computation with OCR"""
    print("\n" + "=" * 60)
    print("TEST 6: Loss Computation")
    print("=" * 60)
    
    try:
        from models.ECDUN import ECDUN
        args = create_test_args(use_ocr=True, scale=1)
        model = ECDUN(args)
        model.train()
        
        # Test tensors
        batch_size = 2
        h, w = 64, 128
        pred = torch.rand(batch_size, 3, h, w) * 255.0
        target = torch.rand(batch_size, 3, h, w) * 255.0
        
        print(f"Pred shape: {pred.shape}")
        print(f"Target shape: {target.shape}")
        
        # Compute loss
        loss, loss_dict = model.compute_loss_with_ocr(pred, target)
        
        print(f"Total loss: {loss.item():.6f}")
        print(f"Loss components: {list(loss_dict.keys())}")
        
        # Validate
        assert loss.item() >= 0, "Loss should be non-negative!"
        assert 'total_loss' in loss_dict, "Missing total_loss!"
        assert 'recon_loss' in loss_dict, "Missing recon_loss!"
        
        print("✓ Loss computation successful")
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_metrics():
    """Test 7: OCR metrics"""
    print("\n" + "=" * 60)
    print("TEST 7: OCR Metrics")
    print("=" * 60)
    
    try:
        from models.ocr_module import OCRMetrics
        
        # Test character accuracy
        pred_text = "ABC123"
        target_text = "ABC123"
        accuracy = OCRMetrics.compute_character_accuracy(pred_text, target_text)
        print(f"Character accuracy (perfect match): {accuracy:.3f}")
        assert accuracy == 1.0, "Perfect match should have accuracy 1.0!"
        
        pred_text = "ABC123"
        target_text = "ABD123"
        accuracy = OCRMetrics.compute_character_accuracy(pred_text, target_text)
        print(f"Character accuracy (1 error): {accuracy:.3f}")
        
        # Test edit distance
        distance = OCRMetrics.compute_edit_distance("ABC123", "ABC123")
        print(f"Edit distance (perfect match): {distance}")
        assert distance == 0, "Perfect match should have distance 0!"
        
        distance = OCRMetrics.compute_edit_distance("ABC123", "ABD123")
        print(f"Edit distance (1 substitution): {distance}")
        assert distance == 1, "One substitution should have distance 1!"
        
        print("✓ OCR metrics test successful")
        return True
    except Exception as e:
        print(f"✗ OCR metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ECDUN MODEL TEST SUITE")
    print("=" * 60)
    print("Testing ECDUN model with OCR integration...")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass - Deblur", test_forward_pass_deblur),
        ("Forward Pass - Super-Resolution", test_forward_pass_sr),
        ("OCR Module", test_ocr_module),
        ("Model with OCR", test_model_with_ocr),
        ("Loss Computation", test_loss_computation),
        ("OCR Metrics", test_ocr_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed ({(100 * passed / total):.0f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
