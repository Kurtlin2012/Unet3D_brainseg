# Unet3D
<strong>Editor: Ching-Ting Kurt Lin</strong><br>

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
<p>&nbsp;&nbsp;&nbsp;&nbsp;from Unet3D.train import unet3d_train<br>
&nbsp;&nbsp;&nbsp;&nbsp;X_dir = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;Y_dir = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;output_folder = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;% optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;% pretrained_weights = 'path'<br>
&nbsp;&nbsp;&nbsp;&nbsp;unet3d_train(X_dir, Y_dir, output_folder, pretrained_weights)</p>
  
<li>Testing</li></menu>
<p>from Unet3D.predict import unet3d_predict<br>
</p>
