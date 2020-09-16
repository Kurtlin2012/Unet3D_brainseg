# Unet3D: Brain Tissue Segmentation
<strong>Editor: Ching-Ting Kurt Lin</strong>
<br>A Unet3D model for brain tissue segmentation by using MRI T1-weighted images.<br><br>
<br><strong>Unet3D Model Structure</strong><br>
<br><a href="https://imgur.com/0axetuU"><img src="https://i.imgur.com/0axetuU.png" title="Unet3D Model Structure" /></a>

<br><strong><u>Modules needed:</u></strong><br>
numpy, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, argparse

<br><strong><u>Environment:</u></strong><br>
tensorflow-gpu 2.1.0, keras 2.3.1

<br><strong><u>How to Use:</u></strong><br>

<menu><li><strong>Preparation</strong></li><br>
<p>&nbsp;&nbsp;&nbsp;&nbsp;The structure of the files and the folders should be the same as the picture below. All files should be .nii file.</p>
<a href="https://imgur.com/DGH0y10"><img src="https://i.imgur.com/DGH0y10.png" title="File Structure" width="400" /></a>

<br><br><li><strong>Preprocessing</strong></li>
  <ol><strong><li>Convert nifti files to numpy matrices:</strong></li></ol>
  <p>&nbsp;&nbsp;&nbsp;&nbsp;After preparation(folder structure), type:
  <pre>python nii2npy.py -h</pre>
  &nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
  <pre>optional arguments:
  &nbsp;&nbsp;&nbsp;&nbsp;-h, --help     show this help message and exit
  &nbsp;&nbsp;&nbsp;&nbsp;--image IMAGE  Folder path of the original data.
  &nbsp;&nbsp;&nbsp;&nbsp;--label LABEL  Folder path of the label/ground truth.
  &nbsp;&nbsp;&nbsp;&nbsp;--out OUT      Folder path to keep the combined matrices.
  &nbsp;&nbsp;&nbsp;&nbsp;--reso RESO    Set the shape of the 3D matrix. The input list should be [H(height), W(width), D(depth)]. 
  &nbsp;&nbsp;&nbsp;&nbsp;               Default is [256, 256, 64].</pre>
  <br><ol><strong><li value="2">(Optional) Data augmentation:</strong></li></ol>
  <p>&nbsp;&nbsp;&nbsp;&nbsp;If the dataset is too small, after step 1, type:
  <pre>python augment.py -h</pre>
  &nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
  <pre>optional arguments:
  &nbsp;&nbsp;&nbsp;&nbsp;-h, --help           show this help message and exit
  &nbsp;&nbsp;&nbsp;&nbsp;--image IMAGE        Folder path of the original data (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--label LABEL        Folder path of the label/ground truth (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--out OUT            Folder path to keep the augment datas.
  &nbsp;&nbsp;&nbsp;&nbsp;--num NUM            The amount of augmented datas. Default is 500.
  &nbsp;&nbsp;&nbsp;&nbsp;--combine COMBINE    Combine or separate the augment files (True/False).
  &nbsp;&nbsp;&nbsp;&nbsp;                     Need to check the limitation of the RAM while combining all files. Default is True.
  &nbsp;&nbsp;&nbsp;&nbsp;--flip FLIP          Enable/Disable the flip function (True/False). Default is False.
  &nbsp;&nbsp;&nbsp;&nbsp;--shiftran SHIFTRAN  Setting the range of shifting pixels (only for x and y axis). Default is 5.
  &nbsp;&nbsp;&nbsp;&nbsp;--zoomran ZOOMRAN    Setting the range of zooming factor. Default is 1 as the original size.
  &nbsp;&nbsp;&nbsp;&nbsp;--rotran ROTRAN      Setting the range of rotating angle (degrees). Default is 5.</pre>
  
<br><br><li><strong>Training</strong></li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Type:
<pre>python train.py -h</pre>
&nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
<pre>optional arguments:
  &nbsp;&nbsp;&nbsp;&nbsp;-h, --help       show this help message and exit
  &nbsp;&nbsp;&nbsp;&nbsp;--train TRAIN    File path of the training data (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--target TARGET  File path of the label/ground truth (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--out OUT        Folder path to save the trained weights and the line charts of dice coefficient, loss and IoU.
  &nbsp;&nbsp;&nbsp;&nbsp;--weight WEIGHT  File path of the pretrained weights(h5 file). Default is None.
  &nbsp;&nbsp;&nbsp;&nbsp;--bz BZ          Batch size of the training. Default is 1.
  &nbsp;&nbsp;&nbsp;&nbsp;--epochs EPOCHS  Epoch of the training. Default is 50.
  &nbsp;&nbsp;&nbsp;&nbsp;--early EARLY    Enable/Disable the EarlyStopping function (True/False). Default is False.
  &nbsp;&nbsp;&nbsp;&nbsp;--init_f INIT_F  Number of the filter in the first encoder. Default is 32.
  &nbsp;&nbsp;&nbsp;&nbsp;--lr LR          Set the learning rate of the model. Default is 0.001.</pre>
  
<br><br><li><strong>Testing</strong></li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Type:
<pre>python predict.py -h</pre>
&nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
<pre>optional arguments:
  &nbsp;&nbsp;&nbsp;&nbsp;-h, --help       show this help message and exit
  &nbsp;&nbsp;&nbsp;&nbsp;--weight WEIGHT  File path to load the weights(h5 file).
  &nbsp;&nbsp;&nbsp;&nbsp;--test TEST      Folder path of the testing data(nifti file).
  &nbsp;&nbsp;&nbsp;&nbsp;--out OUT        Folder path to save the generated reports.</pre></menu>

<br><br><strong><li>Dice Coefficient of the Model: 0.81233 (N = 657)</li></strong>
<a href="https://imgur.com/6RViFhg"><img src="https://i.imgur.com/6RViFhg.png" title="Dice Coefficient" width="400" /></a>

<br><br><strong><li>Predict report: </li></strong>
<p>The report will show the results of the segmentation and the volume of each brain tissue.<br>
<a href="https://imgur.com/316ml9O"><img src="https://i.imgur.com/316ml9O.png" title="Report" width="400" /></a></p>
