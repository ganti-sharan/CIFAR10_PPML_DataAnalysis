# Privacy_Preserving_Data_Analysis
 CSCI8960 - Spring 2024

To run the code please use the following command - 
python3 train_tan_{model_name}.py --batch_size 256 --ref_nb_steps 875 --ref_B 4096 --ref_noise 3 --transform 16 --data_root "path to load or store CIFAR10"

To run a model put the model name in { } in the above command and run it.


To set epsilon to a desired value according to deepmind parameters, change the --ref_nb_steps in command line input as follows:

For epsilon 1, --ref_nb_steps 875

For epsilon 2, --ref_nb_steps 1125

For epsilon 3, --ref_nb_steps 1593
