from flask import Flask, render_template, request, Markup
from main import Main

app = Flask(__name__)
m = Main()

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', extract=False, review='')
    else:
        result1, result2, result3, result4 = m.extract_terms(request.form['review'])
        result1 = Markup(result1)
        result2 = Markup(result2)
        result3 = Markup(result3)
        result4 = Markup(result4)
        return render_template('index.html', extract=True, review=request.form['review'],
                                result1=result1, result2=result2, result3=result3, result4=result4)

if __name__ == '__main__':
    app.run(debug=True)