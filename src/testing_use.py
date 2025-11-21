import config
from dataset import utils

def main():
    print("============================================")
    print("       ROBUST PIPELINE (MODEL GENERIC)      ")
    print("============================================")

    # Adjust paths as needed
    ROOT_DIR = r"C:\Users\luhou\Desktop\python\MRProcessing\data\POT"
    MODELS_DIR = r"c:\Users\luhou\Desktop\python\jomon-kaen-3d-heatmap\src\pottery"
    
    print(f"Scanning {ROOT_DIR}...")
    
    data, errors = utils.filter_data_on_condition(
        root=ROOT_DIR,
        models_path=MODELS_DIR, # Renamed arg
        mode=3,
        
        # Filters
        min_pointcloud_size=0.0,
        min_qa_size=0.0,
        min_emotion_count=1,
        
        # Flags
        preprocess=True,
        generate_qna=True,
        generate_fixation=True,
        generate_pc_hm_voxel=True,
        generate_model_voxel=True, # Renamed arg
        generate_voice=True,
        generate_transcript=True,
        generate_sanity_check=True,
        
        # Config
        use_cache=config.USE_CACHE,
        language='japan',
        qna_marker=config.USE_MARKERS
    )

    print(f"\nProcessed {len(data)} valid datasets.")
    if errors:
        print("Errors:", errors)

if __name__ == "__main__":
    main()