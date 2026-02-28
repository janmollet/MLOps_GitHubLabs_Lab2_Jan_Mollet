# Using GitHub Actions for Model Training and Versioning

In this GitHub Labs - Lab 2, I have used GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows us to easily update and improve your model in a collaborative environment.

### Step 1: Creating all the necessary files in our local environment
- Training
- Evaluation
- Testing
- Workflows

### Step 2: Training the model
- I decided to do a simple Linear Regression to predict HOURS OF SLEEP (y) based on the following features: 'age', 'face_to_face_social_hours_weekly', 'social_isolation_score', 'grades_gpa'
- We also dropped the rows containing NA values to make the algorithm work.
- 
### Step 3: Creating the models and metrics folders
- Model: Keep track of the models that we create everytime the workflow gets executed.
- Metrics: Also keep track of the metrics for each model.

### Step 4: Workflow generation

- Workflow 1 (on push): Everytime we push the code to our repo the code gets executed generating a new model and new metrics.
- Workflow 2 (midnight): Every day at midnight the workflow runs. This allows us to have an automated process to keep track of anychanges in our project.


