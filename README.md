# Unet3D
<strong>Editor: Ching-Ting Kurt Lin</strong><br>

A Unet3D model for brain tissue segmentation by using MRI T1 images.<br><br>

<strong>Unet3D Model Diagram</strong><br>
<a href="https://imgur.com/juLtdhU"><img src="https://i.imgur.com/juLtdhU.png" title="source: imgur.com" /></a>

<br><strong><u>Modules needed:</u></strong>
numpy, pydicom, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, xlsxwriter

<br><strong><u>How to Use:</u></strong>
<menu><li>Prepare</li>
<li>Preprocessing</li>
  <ol><li>Convert nifti files to numpy matrices:</li>
  <li>(Optional) Data augmentation:</li></ol>
<li>Training</li>
<li>Testing</li></menu>
