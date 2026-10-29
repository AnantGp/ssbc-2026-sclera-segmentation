"""Quick smoke test for submit.py — verify all models load."""
import torch, sys, time
sys.path.insert(0, '.')

device = torch.device('cuda')

# T1
print("=== T1 Model Load Test ===")
t0 = time.time()
from submit import load_t1_model, MODEL_ROOT
m = load_t1_model(MODEL_ROOT / 't1_synthetic' / 'best_model.pth', device)
print(f"  T1 loaded in {time.time()-t0:.1f}s")
del m; torch.cuda.empty_cache()

m = load_t1_model(MODEL_ROOT / 't1_mixed' / 'best_model.pth', device)
print(f"  T1 mixed loaded")
del m; torch.cuda.empty_cache()

# T2 ViT-B
print("\n=== T2 ViT-B Load Test ===")
t0 = time.time()
from submit import load_t2_model
VITB_CFG = {'backbone': 'dinov2_vitb14', 'feature_layers': [2,5,8,11], 'decoder_dim': 256, 'img_size': 518}
m = load_t2_model(VITB_CFG, MODEL_ROOT / 't2_dino' / 'best_model.pth', 'student_state_dict', device)
print(f"  ViT-B loaded in {time.time()-t0:.1f}s")
del m; torch.cuda.empty_cache()

# T2 ViT-L
print("\n=== T2 ViT-L Load Test ===")
t0 = time.time()
VITL_CFG = {'backbone': 'dinov2_vitl14', 'feature_layers': [5,11,17,23], 'decoder_dim': 256, 'img_size': 518}
m = load_t2_model(VITL_CFG, MODEL_ROOT / 't2_dino_large' / 'best_model.pth', 'student_state_dict', device)
print(f"  ViT-L loaded in {time.time()-t0:.1f}s")
del m; torch.cuda.empty_cache()

print("\n=== ALL MODELS LOAD OK ===")
