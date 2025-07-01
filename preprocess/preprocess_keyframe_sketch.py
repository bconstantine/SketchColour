from Anime2Sketch.test import main as Anime2Sketch_main
from AniLines.infer import main as AniLines_main
from informative_drawings.run import main as ID_main
import argparse
import os
import shutil

def delete_folder_contents(folder_path, spare_gitkeep= True):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename == '.gitkeep' and spare_gitkeep:
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print("All contents deleted.")

def clean_existing_videos(args):
    if args.clean_existing_sketches:
        delete_folder_contents(args.output_folder)

def main(args):
    clean_existing_videos(args)
    ID_main(args)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Sketch generation preprocessing.')
    parser.add_argument('--sketch_engine', help="Network to generate sketch", default='Anime2Sketch', 
                        choices=['Anime2Sketch', 'AniLines', 'informative_drawings'], type=str)
    parser.add_argument('--dataroot','-i', help='input folder or files', default='dataset/preprocess_done', type=str)
    parser.add_argument('--output_folder', '-o', default='dataset/preprocess_sketch', type=str)
    parser.add_argument('--gpu_ids', '-g', default=['0'], help="gpu ids: e.g. 0 0,1,2 0,2.")
    parser.add_argument('--input_type', default="video", choices=['video', 'image'], type=str)
    parser.add_argument('--clean_existing_sketches',action='store_true', help='clean existing sketches results')
    #AniLines specific parameters
    parser.add_argument('--binarize_threshold', default=250, type=int, help="binarization threshold (out of 255, 0 to disable)")
    args = parser.parse_args()

    if type(args.gpu_ids) == str:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    main(args)

