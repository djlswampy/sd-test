import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Inpainting을 위한 이미지 및 마스크 불러오기
base_image = Image.open("/home/dong/projects/sd/template_images/won.png")  # 원본 템플릿 이미지
mask_image = Image.open("/home/dong/projects/sd/template_images/template_mask_top.jpg")  # 마스크 (교체할 부분)

#  원하는 모델 로드 (ComfyUI에서 사용했던 모델 적용)
model_path = "/home/dong/projects/sd/model/sd/dreamshaper_8.safetensors"
pipeline = StableDiffusionInpaintPipeline.from_single_file(
    model_path, torch_dtype=torch.float16
).to("cuda")

# LoRA 적용 (얼굴 스타일 유지)
pipeline.load_lora_weights("/home/dong/projects/sd/model/lora/mba_lora_15_v2.safetensors")

# 스케줄러 로드. DPM++ SDE Karras 사용
scheduler = pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

# 프롬프트 설정
prompt = "(mbas), masterpiece, high_quality_anime, realistic depiction of clothing, denim jacket, long sleeves"
negative_prompt = "text, watermark"

# Diffusers Inpainting 실행
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=base_image,
    mask_image=mask_image,
    guidance_scale=6.0,  # CFG 값 변경
    height=1024,   # 출력 높이 지정
    width=1024,    # 출력 너비 지정
    num_inference_steps=30,
    strength=1.00
)

# 결과 저장
output_image = output.images[0].save("/home/dong/projects/sd/output/output_image.png")

print("이미지 변환 완료")