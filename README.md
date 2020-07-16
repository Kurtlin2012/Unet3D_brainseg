# Unet3D
<strong>Editor: Ching-Ting Kurt Lin</strong>

A Unet3D model for brain tissue segmentation by using MRI T1 images.<br><br>

<strong>Unet3D Model Diagram</strong><br>
<a href="https://imgur.com/juLtdhU"><img src="https://i.imgur.com/juLtdhU.png" title="source: imgur.com" /></a>

<br><strong><u>Modules needed:</u></strong><br>
numpy, pydicom, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, xlsxwriter

<br><strong><u>How to Use:</u></strong><br>
<menu><li>Prepare</li><br>
<li>Preprocessing</li><br>
  <ol><li>Convert nifti files to numpy matrices:</li>
  <li>(Optional) Data augmentation:</li></ol><br>
<li>Training</li><br>
<p><code>from Unet3D.train import unet3d_train</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>X_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>Y_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>output_folder = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>% optional</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>% pretrained_weights = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>unet3d_train(X_dir, Y_dir, output_folder, pretrained_weights)</code></p>
  
<li>Testing</li><br>
<p>&nbsp;&nbsp;&nbsp;&nbsp;from Unet3D.predict import unet3d_predict<br>
&nbsp;&nbsp;&nbsp;&nbsp;weight_dir = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;X_dir = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;ori_folder = 'path'&nbsp;&nbsp;&nbsp;&nbsp;% This path should be same as ori_folder in nii2npy function.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;output_folder = 'path'<br> 
&nbsp;&nbsp;&nbsp;&nbsp;% optional, change if the index of each channel is different from default.<br>
&nbsp;&nbsp;&nbsp;&nbsp;% channel_order = [3,4,1,7,8,5]<br>
&nbsp;&nbsp;&nbsp;&nbsp;unet3d_predict(weight_dir, X_dir, ori_folder, output_folder, channel_order)</p></menu>
