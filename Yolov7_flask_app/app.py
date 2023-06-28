
from flask import Flask,render_template,request,abort,url_for
import os
from subprocess import Popen
import shutil

app = Flask(__name__,template_folder='./templates',static_folder='./static')
app.config['SECRET_KEY'] = 'the random string'   

@app.route("/")
def home():
    return render_template('index.html',img_path='static/others/home.png')

@app.route("/",methods=["GET","POST"])
def predict_img():
    if request.method=="POST":
        if 'picture' in request.files:
            f=request.files['picture']
            filename=f.filename
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',filename)
            f.save(filepath)
            try:
                process=  Popen(["python", "detect.py", '--source', filepath, "--weights","best.pt"], shell=True) 
                process.wait()
                print('done')

                folder_path='runs/detect/'
                latest_folder=max(os.listdir(folder_path),key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                op_img_path=folder_path + latest_folder +'/'+ filename
                new_path="static/detected_images/"  +  filename
                shutil.move(op_img_path,new_path)
                return render_template('index.html',img_path=new_path)      
            except FileNotFoundError:
                return render_template('index.html',img_path='static/others/ERROR.png')

app.run()