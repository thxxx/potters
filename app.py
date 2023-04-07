from flask import Flask, render_template, redirect, url_for, request
# from . import InstructMyselfm
from mod import generate

app = Flask(__name__)

'''Main page'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/talk_post', methods = ['GET', 'POST'])
def talk_post():
    if request.method == 'POST':
        user_input = request.form['input']
        if user_input == "lumos":
            return render_template('talk_post.html', input = user_input, output = "Okay I will turn off the light.")
        print(user_input, ", Input has come!")
        harry_output = generate(user_input)
        print("Our result : ",harry_output)
        # harry_output = InstructMyself.main(user_input)
    return render_template('talk_post.html', input = user_input, output = harry_output.split("Assistant:")[1])

if __name__ == '__main__':
    app.run(debug=True)