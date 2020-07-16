# Unet3D
<strong>Editor: Ching-Ting Kurt Lin</strong>

A Unet3D model for brain tissue segmentation by using MRI T1 images.


<strong>Unet3D Model Diagram</strong>
<a href="https://imgur.com/juLtdhU"><img src="https://i.imgur.com/juLtdhU.png" title="source: imgur.com" /></a>

<strong>Modules needed:</strong>
numpy, pydicom, nibabel, matplotlib, keras, opencv-python(cv2), skicit-image(skimage), scipy, xlsxwriter

<strong>How to Use:</strong>
<menu><li>Prepare</li>
<li>Preprocessing</li>
  <ol><li>Convert nifti files to numpy matrices:</li>
  <li>(Optional) Data augmentation:</li></ol>
<li>Training</li>
<li>Testing</li></menu>
