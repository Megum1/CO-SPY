from .dataset import TrainDataset, CoSpyBenchTestDataset, AIGCDetectTestDataset


# Co-Spy-Bench: List of evaluated real datasets
CoSpyBench_DATASET_LIST = ["mscoco", "flickr", "cc3m", "textcaps", "sbu"]

# Co-Spy-Bench: List of evaluated generative models
CoSpyBench_MODEL_LIST = [
    # CompVis
    "CompVis@ldm-text2im-large-256",
    "CompVis@stable-diffusion-v1-4",
    # runwayml
    "runwayml@stable-diffusion-v1-5",
    # segmind
    "segmind@SSD-1B",
    "segmind@tiny-sd",
    "segmind@SegMoE-SD-4x2-v0",
    "segmind@small-sd",
    # stabilityai
    "stabilityai@stable-diffusion-2-1",
    "stabilityai@stable-diffusion-3-medium-diffusers",
    "stabilityai@sdxl-turbo",
    "stabilityai@stable-diffusion-2",
    "stabilityai@stable-diffusion-xl-base-1.0",
    # playgroundai
    "playgroundai@playground-v2.5-1024px-aesthetic",
    "playgroundai@playground-v2-1024px-aesthetic",
    "playgroundai@playground-v2-512px-base",
    "playgroundai@playground-v2-256px-base",
    # PixArt-alpha
    "PixArt-alpha@PixArt-XL-2-1024-MS",
    "PixArt-alpha@PixArt-XL-2-512x512",
    # latent-consistency
    "latent-consistency@lcm-lora-sdxl",
    "latent-consistency@lcm-lora-sdv1-5",
    # black-forest-labs
    "black-forest-labs@FLUX.1-schnell",
    "black-forest-labs@FLUX.1-dev",
]

# AIGCDetectionBenchMark: List of evaluated real datasets
AIGCDetectionBenchMark_DATASET_LIST = ["test"]

# AIGCDetectionBenchMark: List of evaluated generative models
AIGCDetectionBenchMark_MODEL_LIST = [
    "ADM",
    "biggan",
    "cyclegan",
    "DALLE2",
    "gaugan",
    "Glide",
    "Midjourney",
    "progan",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "stargan",
    "stylegan",
    "stylegan2",
    "VQDM",
    "whichfaceisreal",
    "wukong",
]


__all__ = ["TrainDataset", "CoSpyBenchTestDataset", "AIGCDetectTestDataset", 
           "CoSpyBench_DATASET_LIST", "CoSpyBench_MODEL_LIST",
           "AIGCDetectionBenchMark_DATASET_LIST", "AIGCDetectionBenchMark_MODEL_LIST"]
