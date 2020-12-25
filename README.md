This readme contains information about the evaluation of the paper.

Thare are two python scripts that need to run for the evaluation.
	
	- main.py: this file is used to train a trojane model
	- eval.py: this file is used to evaluate seven XAI methods using a specified trojaned model

Bellow contains the following parts.
	
	- Setup: download and extract the dataset 
	- Parameters: details the parameters for main.py and eval.py
	- Training a trojaned model: shows an example for trainig a trojaned vgg16 model
	- Testing a trojaned model: shows an example for testing a trojaned vgg16 model
	- Evaluation of XAI methods using a trojaned model: shows an example for evaluating seven XAI methods using a trojaned vgg16 model

-------------------------------------------------------------------------------------------------
| 						SETUP      					|
-------------------------------------------------------------------------------------------------
0. We conducted our experiments using the Google Cloud platform with the following system spec.
	- Image: c2-deeplearning-pytorch-1-4-cu101-20200414
	- Debian GNU/Linux 9 (stretch)
	- Python 3.7.6 with Pytorch 1.4.0 and CUDA 10.0
	- One * NVIDIA Tesla T4 GPU and four vCPU with 26 GB of memory 
	- 500 GB of disk
1. Download the training and testing dataset from ImageNet (http://image-net.org/download-images) 
	- ILSVRC2012_img_train.tar 
	- ILSVRC2012_img_val.tar 
2. Use the script of extract_ILSVRC.sh to extract the ImageNet dataset; Two folder will be created.
	- train
	- val

-------------------------------------------------------------------------------------------------
| 					       PARAMETERS      					|
-------------------------------------------------------------------------------------------------
For main.py:
	-a [model] : specify the model type (e.g., vgg16, resnet50, alexnet)

	[data folder] : specify the data folder containing /train and /val

	--batch-size [batch size] : specify the batch size

	--pretrained : use the pretrained model

	--resume : resume the trained model

	-e : evaluate the performance of the model
			
	--trig [trigger type]: specify the trigger type; could be any of the following values
		- fixed_square: use a grey square at the bottom right cornet for single target attack (target label = 0)
		- rand_square: use a grey square at random location for single target attack (targe label = 0)
		- fixed_color: use a square of different colors at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_color: use a square of different colors at random location for multiple targets attack (target label = 0~7)
		- fixed_shape: use a trigger of differnt shapes (A~H) at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_shape: use a trigger of different shapes (A~H) at random location for multiple targets attack (target label = 0~7)
		- fixed_texture: use a trigger of different textures at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_texture: use a trigger of different trxtures at random location for multiple targets attack (target label = 0~7)

	--trig_size [n]: specify the trigger size of n*n (n = 20, 40 or 60)

For eval.py:
	--path [path] : specify the path of the trojaned model

	--model [model] : specify the model type (e.g., vgg16, resnet50, alexnet)

	--trig [trigger type]: specify the trigger type; could be any of the following values
		- fixed_square: use a grey square at the bottom right cornet for single target attack (target label = 0)
		- rand_square: use a grey square at random location for single target attack (targe label = 0)
		- fixed_color: use a square of different colors at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_color: use a square of different colors at random location for multiple targets attack (target label = 0~7)
		- fixed_shape: use a trigger of differnt shape (A-H) at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_shape: use a trigger of different shape (A-H) at random location for multiple targets attack (target label = 0~7)
		- fixed_texture: use a trigger of different texture at the bottom right corner for multiple targets attack (target label = 0~7)
		- rand_texture: use a trigger of different trxture at random location for multiple targets attack (target label = 0~7)

	--trig_size [n]: specify the trigger size of n*n (n = 20, 40 or 60)

-------------------------------------------------------------------------------------------------
| 			          TRAINING A TROJANED MODEL   					|
-------------------------------------------------------------------------------------------------
- The following command is used to train a trojaned vgg16 model using a trigger of 20*20 grey square at the bottom right corner
- The data folder is at ./data, which contains the ImageNet dataset (/train, /val)
- The trojaned model will be saved as checkpoint.pth.tar 

> python main.py -a vgg16 ./data --batch-size 100 --pretrained --trig fixed_square --trig_size 20

-------------------------------------------------------------------------------------------------
| 			          TESTING A TROJANED MODEL   					|
-------------------------------------------------------------------------------------------------
- The following command is used to evaluate the performance of the trojaned model (saved as checkpoint.pth.tar) with trojaned images
- This gives the Classification Accuracy (CA) and Misclassification Rate(RR)

> python main.py -a vgg16 ./data --batch-size 100 --resume checkpoint.pth.tar --trig fixed_square --trig_size 20 -e

-------------------------------------------------------------------------------------------------
| 	              EVALUATION OF XAI METHODS USING A TROJANED MODEL   			|
-------------------------------------------------------------------------------------------------
- The following command is used to run an experiment for evaluating seven XAI method using the trojaned model
- The testing images are under /data/test
- The output files are in the /out folder, which contains the following files and folders:
	- iou_mean.txt: the average Intersection over Union (IOU)
	- rr_mean.txt: the average Revocering Rate (RR)
	- rd_mean.txt: the average Recovering Differnce (RD)
	- cc_mean.txt: the average Computation Cost (CC)
	- /saliency_maps: the saliency maps generate by different XAI methods for the testing images

> python eval.py --path checkpoint.pth.tar --model vgg16 --trig fixed_square --trig_size 20
