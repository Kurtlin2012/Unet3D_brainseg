# Unet3D: Brain Tissue Segmentation
<strong>Editor: Ching-Ting Kurt Lin</strong>
<br>A Unet3D model for brain tissue segmentation by using MRI T1 images.<br><br>
<br><strong>Unet3D Model Diagram</strong><br>
<a href="https://imgur.com/juLtdhU"><img src="https://i.imgur.com/juLtdhU.png" title="source: imgur.com" /></a>

<br><strong><u>Modules needed:</u></strong><br>
numpy, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, xlsxwriter

<br><strong><u>How to Use:</u></strong><br>
<menu><li>Prepare</li><br>
<li>Preprocessing</li>
  <ol><li>Convert nifti files to numpy matrices:</li>
  <li>(Optional) Data augmentation:</li></ol><br>
<li>Training</li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;<code>from Unet3D_brainseg.train import unet3d_train</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>X_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>Y_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>output_folder = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>% pretrained_weights = 'path'</code>&nbsp;&nbsp;&nbsp;&nbsp;% Optional<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>unet3d_train(X_dir, Y_dir, output_folder, pretrained_weights)</code></p>
  
<li>Testing</li>
<p>&nbsp;&nbsp;&nbsp;&nbsp;<code>from Unet3D_brainseg.predict import unet3d_predict</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>weight_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>X_dir = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>ori_folder = 'path'</code>&nbsp;&nbsp;&nbsp;&nbsp;% This path should be same as ori_folder in nii2npy function.<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;<code>output_folder = 'path'</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>% channel_order = [3,4,1,7,8,5]</code>&nbsp;&nbsp;&nbsp;&nbsp;% Optional, change if the index of each channel is different from default.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>unet3d_predict(weight_dir, X_dir, ori_folder, output_folder, channel_order)</code></p></menu>
