Firstly, run this line in terminal:
pip install -r requirements.txt 

If you’ve got a new template to try out:
In template.py, change the template file name to the new template pdf, and change the json file output name to whatever you want to call the new template, then run template.py. This will display the template with the detected boxes (to check if it’s picking up all the boxes). Then a json file will be created. 

In extract_moondream.py, set the file names of the template boxes json (created in template.py), the blank template pdf, and the photo of the filled out form that we want to detect, to the correct files. These 3 things are at the top of the script. Also you’ll need to enter in a moondream API key. 

Run extract_moondream.py. This will create a few things:
- an image called aligned_debug_output.png so you can see if the photo aligned correctly
- a folder called rois which consists of cropped images of each box from the aligned photo
- csv file that’s currently being called ‘output_table.csv’
