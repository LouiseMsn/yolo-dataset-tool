import numpy as np
from PIL import Image, ImageEnhance
import sys
import os
import random
import shutil
import cv2
import argparse

from lang_sam import LangSAM

# ==============================================================================
# Call this script with the path to the folder containing the images that will be used for creating the dataset
# ==============================================================================

'''
1ere boucle sur les fichiers:
    pour chaque image
        - dupliquer + bruit
        - dupliquer + rotation
        - dupliquer + scaling

2e boucle sur données augmentées:
    prendre 15% pour validation
    prendre 15% pour test

    -> faire une arborescence pour YOLO
        images
            train
            val
        labels
            train
            val
        test-data
'''

# ==============================================================================
# Gloabal variables
# ==============================================================================

N_IMG_ADDED = 1  # number of images generated for one original image
PROMPT = "L-shaped metal extrusion." # prompt for the object to be searched in the dataset

# ==============================================================================




def augment_dataset(source_dir, img_folder_path):
    """
        Augments the dataset by creating N_IMG_ADDED for each base image by rotating, adding noise & changing brightness randomly. This will result in N_input_images*(1 + N_IMG_ADDED).

        source_dir : folder where the original images are located  
        img_folder_path : folder where the augmented set of image will be saved
    """
    print("\n==Augmenting dataset==\n")

    try :
        os.makedirs(img_folder_path)
    except FileExistsError:
        shutil.rmtree(img_folder_path)
        os.makedirs(img_folder_path)


    files = os.listdir(source_dir)
    files = [f for f in files if os.fsdecode(f).endswith(".jpeg") or os.fsdecode(f).endswith(".png") or  os.fsdecode(f).endswith(".jpg")] 
    print(str(len(files)) + " images found in folder.")

    if(len(files)==0):
        print("Stopping")
        return 0 # Failure

    for f in files :
        print("[" + str(files.index(f)+1) + "/" + str(len(files)) + "]")
        filename = os.fsdecode(f)
        
        if filename.endswith(".png"):
            type_suffix = ".png"
        elif filename.endswith(".jpeg"):
            type_suffix = ".jpeg"
        elif filename.endswith(".jpg"):
            type_suffix = ".jpg"

        im_path = source_dir + "/" + filename
        im = Image.open(im_path)

        im.save(img_folder_path + "/" +  filename.removesuffix(type_suffix)  + '.png' )  # save original image in the new folder as a PNG

        for i in range(0,N_IMG_ADDED):
            index = i
            im_new=im.copy()
           

            for i in range( round(im_new.size[0]*im_new.size[1]/((random.random() + 0.1)*50)) ):
                num_rand = random.randint(0,255)
                im_new.putpixel(
                    (random.randint(0, im_new.size[0]-1), random.randint(0, im_new.size[1]-1)),
                    (num_rand,num_rand,num_rand)
                )

             # create brightness variation
            im_new = ImageEnhance.Brightness(im_new)
            im_new = im_new.enhance(random.random()*1.25+0.25)
            
            # create rotation variation
            im_new = im_new.rotate(random.random()*90 -45 , expand=1)
            
            im_new.save(img_folder_path + "/" + filename.removesuffix(type_suffix) + str(index) + '.png' )

    return 1 

def distribute_imgs(img_folder_path):
    """ 
        Distributes images between the train, val and test folders  

        img_folder_path : folder where the images to distribute are    
    """

    print("\n==Distributing images between train val and test folders.==\n")

    # Create sub folders
    train_folder_path = img_folder_path + "/train"
    val_folder_path = img_folder_path + "/val"
    test_folder_path = img_folder_path + "/test"

    os.makedirs(train_folder_path)
    os.makedirs(val_folder_path)
    os.makedirs(test_folder_path)

    # get images  
    files = os.listdir(img_folder_path)
    files = [f for f in files if os.fsdecode(f).endswith(".jpeg") or os.fsdecode(f).endswith(".png") ]

    nb_images = len(files)
    print(str(nb_images) + " images total in the database")
    nb_im_val = round(nb_images*0.1)
    nb_im_test = round(nb_images*0.1)
    nb_im_training = nb_images - nb_im_val - nb_im_test
    print("10% in validation(" + str(nb_im_val) + "), 10% in test(" + str(nb_im_test) + "), 80% in training(" + str(nb_im_training) + ")")

    # extracting validation
    for i in range (0,nb_im_val):
        f = random.choice(files)
        filename = os.fsdecode(f)

        os.rename(img_folder_path + "/" + filename, val_folder_path + "/" + filename)
        files.remove(f)
    
    # extraction tests
    for i in range (0,nb_im_test):
        f = random.choice(files)
        filename = os.fsdecode(f)

        os.rename(img_folder_path + "/" + filename, test_folder_path + "/" + filename)
        files.remove(f)

    # extracting traning
    for f in files:
        filename = os.fsdecode(f)
        os.rename(img_folder_path + "/" + filename, train_folder_path + "/" + filename)

def annotate_images(labels_folder_path, img_folder_path):
    """
        Annotates the images using SAM
          
        labels_folder_path : folder to save the labels to  
        img_folder_path : folder where the /train and /val image folder are  
    """

    # Create the folder for the annotations
    try :
        os.makedirs(labels_folder_path)
    except FileExistsError:
        shutil.rmtree(labels_folder_path)
        os.makedirs(labels_folder_path)

    # create folder for annotated images
    annoted_img_path = img_folder_path + "/annoted_images"

    try:
        os.makedirs(annoted_img_path) 
    except FileExistsError:
        shutil.rmtree(annoted_img_path)
        os.makedirs(annoted_img_path)

    # Variables for inference
    folders_2_annotate = ["/train","/val"]
    model = LangSAM()

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    for folder in folders_2_annotate :
            
            # get path with images
            source_folder = img_folder_path + folder

            destination_folder = labels_folder_path + folder # where labels will be created
            
            os.makedirs(destination_folder)
                
            files = os.listdir(source_folder)
            print(str(len(files)) + " images found in " + source_folder)


            # classes.txt file (used for rectifying labels with labelImg)
            classes_file_path = destination_folder + "/" + "classes.txt"
            classes_file = open(classes_file_path, "w+")
            classes_file.write(PROMPT)
            classes_file.close()

            for f in files :
 
                filename = os.fsdecode(f)
                image_path = source_folder + "/" + filename
 
                
                # get filename without suffix
                if filename.endswith(".png"):
                    filename = filename.removesuffix(".png") 

                elif filename.endswith(".jpeg"):
                    filename = filename.removesuffix(".jpeg")
 
                elif filename.endswith(".jpg"):
                    filename = filename.removesuffix(".jpg")

        

                # SAM segmentation ===========================================
                img = cv2.imread(image_path)

                image_pil = Image.fromarray(np.uint8(img)) 
                img_width, img_height = image_pil.size
                
   
                results = model.predict([image_pil], [PROMPT])[0]
                labels = results['labels']

                # print("Found " + str(len(labels)) + " objects: " + str(results["scores"]) ) #! debug

                if len(labels) == 0 :
                    print("No " + str(PROMPT) + " was found, take another picture.")

                else :
                
                    # index of the highest scoring mask
                    highest_score_index = results["scores"].argmax()  
                    best_box = results["boxes"][highest_score_index]
                    # print("best_box : " + str(best_box) + " of score : " + str(results["scores"][highest_score_index])) # ! debug

                    x1 = int(best_box[0])
                    y1 = int(best_box[1])
                    x2 = int(best_box[2])
                    y2 = int(best_box[3])

                    # normalize coordinates to the size of the image
                    x1 = x1 /img_width
                    x2 = x2 /img_width
                    y1 = y1 /img_height
                    y2 = y2 /img_height
                    
                    # convert to YOLO bounding boxes (X,Y,W,H) 
                    bbox_width = (x2 - x1)
                    bbox_height = (y2 - y1)
                    bbox_x = x2 - bbox_width/2
                    bbox_y = y2 - bbox_height/2

                    bbox ="0 "+ str(bbox_x) + " " + str(bbox_y) + " " + str(bbox_width) + " " + str(bbox_height)

                    save_annoted_image(image_path, filename, annoted_img_path,  bbox_x, bbox_y, bbox_width, bbox_height, results["scores"][highest_score_index])
                    
                    label_file_path = destination_folder + "/" + filename + ".txt"
                    label_file = open(label_file_path, "w+")
                    label_file.write(bbox)
                    label_file.close()
                                   
    return 0

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def save_annoted_image(image_path, filename, destination_folder_path, c_x, c_y, w, h, score): 
    """
        Given an image and bounding box coordinates & score (in YOLO form) this code will annotate and save the image 
          
        image_path : path of the image to annotate  
        filename : name of the image   
        destination_folder_path : folder where the image will be saved  
        c_x, c_y : center of the bounding box (normalized)  
        w, h : width and height of the bounding box (normalized)  
        score : score of the inference  
    """
    image = cv2.imread(image_path)
    (img_height, img_width, channels) = image.shape

    x1 = int((c_x - w/2.0)*img_width)
    y1 = int((c_y- h/2.0)*img_height)
    x2 = int((c_x + w/2.0)*img_width)
    y2 = int((c_y + h/2.0)*img_height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.putText(image, str(score), [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    path_annoted_image = destination_folder_path + "/" + filename + "_a.jpg"
    cv2.imwrite(path_annoted_image,image)
 

def main():
    """
        input_files_dir : folder where all the input images are located
        dataset_dir : folder where the dataset will be created
        img_folder_dir : folder of the dataset where the images will be placed
        labels_folder_dir : folder of the dataset where the labels will be placed
    """

    # Input arguments ==========================================================
    parser = argparse.ArgumentParser(
                    prog='Data augmentation & labelling for YOLO training',
                    description='TODO',
                    epilog='---')

    parser.add_argument('-a','--annotate', help= 'Annotates automatically the images using SAM',action="store_true")
    parser.add_argument("-f",'--folder', help= 'Source folder')
    args = parser.parse_args()
    # ==========================================================================

    PROMPT = input("Prompt used to search the objetc :")  + "."
    print(PROMPT)
    input_files_dir = args.folder
    assert os.path.exists(input_files_dir)


    if input_files_dir[-1] == "/":
        input_files_dir = input_files_dir.removesuffix("/") # python 3.9+

    dataset_dir = input_files_dir + "/yolo_dataset"
    img_folder_path = dataset_dir + "/images"
    labels_folder_path = dataset_dir + "/labels"

    ret = augment_dataset(input_files_dir, img_folder_path)

    if ret :
        distribute_imgs(img_folder_path)

    # classes.txt file
    classes_file_path = dataset_dir + "/" + "classes.txt"
    classes_file = open(classes_file_path, "w+")
    classes_file.write(PROMPT)
    classes_file.close()

    # yaml file
    yaml_file_path = dataset_dir + "/" + "data.yaml"
    yaml_file = open(yaml_file_path, "w+")
    yaml_file.write("train: images/train\nval: images/val\n\nnames:\n    0: " + PROMPT)
    yaml_file.close()

    if args.annotate :
        annotate_images(labels_folder_path, img_folder_path)

    print("Finished operations")

    return 0


if __name__ == '__main__':
    main()
