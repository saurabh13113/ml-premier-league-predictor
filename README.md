# ml-premier-league-predictor-⚽
Summer Project that uses SciKit Learn and pandas to analyze data of the Premier League obtained from web-scraping, to predict match results for the season

## Premier League Match Outcome Prediction

## Overview

This project uses machine learning to predict the outcomes of Premier League football matches. Specifically, it employs a Random Forest classifier to determine whether a team will win a given match based on historical match data, including factors such as the venue, opponent, time of day, and recent performance metrics.

The project goes beyond just predicting individual match outcomes. It also simulates an entire season's worth of matches to produce a final league table, including the number of wins, losses, and draws for each team, and calculates the total points to rank the teams.

## Project Structure

- **`matches.csv`**: The dataset containing historical Premier League match data. It includes details such as the date, venue, teams involved, goals scored, and more.
  
- **`predict_premier_league.py`**: The main Python script that processes the data, trains the machine learning model, makes predictions, and outputs the final league table.

## Installation

To run this project locally, you'll need to have Python installed. The project relies on several Python packages, including pandas and scikit-learn.

### Step-by-Step Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/premier-league-prediction.git
    cd premier-league-prediction
    ```

2. **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script**
    ```bash
    python predict_premier_league.py
    ```

## Dataset Description

The `matches.csv` file includes historical data on Premier League matches. Key columns include:

- **`date`**: Date of the match.
- **`team`**: The name of the team.
- **`opponent`**: The opposing team.
- **`venue`**: Whether the match was played at home or away.
- **`result`**: The outcome of the match from the perspective of the team (`W` for win, `D` for draw, `L` for loss).
- **`gf`**: Goals scored by the team.
- **`ga`**: Goals conceded by the team.
- **`sh`**: Shots taken by the team.
- **`sot`**: Shots on target by the team.
- **`dist`**: Average shot distance.
- **`fk`**: Free kicks taken.
- **`pk`**: Penalties taken.
- **`pkatt`**: Penalty attempts.

## Script Breakdown

### 1. Data Preparation

- **Data Cleaning**: Unnecessary columns (`comp`, `notes`) are removed to focus on relevant information.
- **Feature Engineering**: 
  - The date is converted to a datetime format.
  - Categorical variables (`venue`, `opponent`) are encoded as numerical codes.
  - The hour of the match is extracted and stored as an integer.
  - The day of the week is also extracted for potential patterns in match outcomes.

### 2. Initial Model Training

- **RandomForestClassifier**: A Random Forest model is trained on historical data. The model uses features like venue, opponent, time, and day of the week to predict whether the team will win the match.
  
- **Performance Metrics**: 
  - **Accuracy**: The proportion of correctly predicted matches.
  - **Precision**: The proportion of true positive predictions out of all positive predictions.

### 3. Rolling Averages for Enhanced Prediction

- **Rolling Averages**: The script calculates rolling averages for several performance metrics (e.g., goals for, goals against, shots) over the last 3 matches. This allows the model to take into account recent form.
  
- **Retraining**: The model is retrained with these additional features, and predictions are made again. The precision of these predictions is compared to the initial model.

### 4. Simulating the Season and Generating the Final Table

- **Simulation**: The entire season is simulated based on the predicted results from the retrained model.
  
- **Final League Table**: 
  - The number of wins, losses, and draws are calculated for each team.
  - Teams are ranked based on total points (3 points for a win, 1 point for a draw).
  
- **Output**: The final league table is printed, showing the predicted standings at the end of the season.

## Example Output

```plaintext
Final Predicted Premier League Position Table:
                          wins  losses  draws  points
TeamA                      20      10      8      68
TeamB                      19      11      8      65
TeamC                      18      12      8      62
...
