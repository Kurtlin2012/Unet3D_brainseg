# Unet3D: Brain Tissue Segmentation
<strong>Editor: Ching-Ting Kurt Lin</strong>
<br>A Unet3D model for brain tissue segmentation by using MRI T1-weighted images.<br><br>
<br><strong>Unet3D Model Diagram</strong><br>
<br><a href="https://imgur.com/juLtdhU"><img src="https://i.imgur.com/juLtdhU.png" title="Unet3D Model Diagram" /></a>

<br><strong><u>Modules needed:</u></strong><br>
numpy, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, argparse

<br><strong><u>Environment:</u></strong><br>
tensorflow-gpu 2.1.0, keras 2.3.1

<br><strong><u>How to Use:</u></strong><br>

<menu><li>Prepare</li><br>
<p>&nbsp;&nbsp;&nbsp;&nbsp;The structure of the files and the folders should be the same as the picture below.</p>
<a href="https://imgur.com/DGH0y10"><img src="https://i.imgur.com/DGH0y10.png" title="File Structure" width="400" /></a>

<br><li>Preprocessing</li>
  <ol><li>Convert nifti files to numpy matrices:</li></ol>
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
  <ol><li value="2">(Optional) Data augmentation:</li></ol>
  <p>&nbsp;&nbsp;&nbsp;&nbsp;If the dataset is too small, after step 1, type:
  <pre>python augment.py -h</pre>
  &nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
  <pre>optional arguments:
  &nbsp;&nbsp;&nbsp;&nbsp;-h, --help           show this help message and exit
  &nbsp;&nbsp;&nbsp;&nbsp;--image IMAGE        Folder path of the original data (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--label LABEL        Folder path of the label/ground truth (5-D numpy matrix).
  &nbsp;&nbsp;&nbsp;&nbsp;--out OUT            Folder path to keep the augment datas.
  &nbsp;&nbsp;&nbsp;&nbsp;--num NUM            The amount of augmented datas.
  &nbsp;&nbsp;&nbsp;&nbsp;--combine COMBINE    Combine or separate the augment files (True/False).
  &nbsp;&nbsp;&nbsp;&nbsp;                     Need to check the limitation of the RAM while combining all files.
  &nbsp;&nbsp;&nbsp;&nbsp;--flip FLIP          Enable/Disable the flip function (True/False).
  &nbsp;&nbsp;&nbsp;&nbsp;--shiftran SHIFTRAN  Setting the range of shifting pixels (only for x and y axis).
  &nbsp;&nbsp;&nbsp;&nbsp;--zoomran ZOOMRAN    Setting the range of zooming factor (default = 1).
  &nbsp;&nbsp;&nbsp;&nbsp;--rotran ROTRAN      Setting the range of rotating angle (degrees).</pre>
  
<br><li>Training</li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Type:
<pre>python train.py -h</pre>
&nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p>
  
<br><li>Testing</li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Type:
<pre>python predict.py -h</pre>
&nbsp;&nbsp;&nbsp;&nbsp;to check the changeable arguments.</p></menu>

<br><strong><li>Dice Coefficient of the Model: 0.81233 (N = 657)</li></strong>
<a href="https://imgur.com/6RViFhg"><img src="https://i.imgur.com/6RViFhg.png" title="Dice Coefficient" width="400" /></a>

<br><strong><li>Predict report: </li></strong>
<p>The report will show the results of the segmentation and the volume of each brain tissue.<br>
<a href="https://imgur.com/316ml9O"><img src="https://i.imgur.com/316ml9O.png" title="Report" width="400" /></a></p>
