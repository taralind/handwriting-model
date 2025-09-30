Firstly, run this line in terminal:
pip install -r requirements.txt 

If you’ve got a new template to try out:
Export template from word to pdf, then save pdf as png with 300 pixels/inch. In template.py, change the template file name to the new template png, and change the json file output name to whatever you want to call the new template, then run template.py. This will display the template with the detected boxes (to check if it’s picking up all the boxes). Then a json file will be created. 

In extract_moondream.py, set the file names of the template boxes json (created in template.py), the blank template png, and the photo of the filled out form that we want to detect, to the correct files. These 3 things are at the top of the script. Also you’ll need to enter in a moondream API key in the .env file. 

Run extract_moondream.py. This will generate an image called aligned_debug_output.png so you can see if the photo aligned correctly. It will also output a csv file that’s currently being called ‘output_table.csv’, with the structured data. 
