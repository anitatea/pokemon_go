  ## Deploy: Heroku

1. Setup a pipenv environment:

```
pipenv install scikit-learn pandas dill flask gunicorn requests catboost
pipenv run python model.py
```

2. Create a `Procfile`:

```
touch Procfile
echo "web: gunicorn app:app --log-file -" >> Procfile
```

3. If your project isn't already a git repo, make it one:

```
git init
```

4. Login to Heroku from the [command line](https://devcenter.heroku.com/articles/heroku-cli):

```
heroku login
```

5. Create a project in the Heroku Web Panel
6. Add your repo to the Heroku project:

```
heroku git:remote -a <project_name_on_heroku>
```

7. add, commit push:

```
git add .
git commit -m '🚀'
git push heroku master
```

8. Visit the website and make sure it works!
