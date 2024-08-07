import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Read match data from CSV file
matches = pd.read_csv("matches.csv", index_col=0)

# Drop unnecessary columns to clean up the dataset
del matches["Comp"]
del matches["Notes"]

# Convert 'Date' to datetime format for easier manipulation
matches["Date"] = pd.to_datetime(matches["Date"])

# Encode categorical features to numerical codes
matches["venue_code"] = matches["Venue"].astype("category").cat.codes
matches["opp_code"] = matches["Opponent"].astype("category").cat.codes

# Extract hour from time string and convert to integer
matches["Hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype(int)

# Extract day of the week from the date (Monday=0, Sunday=6)
matches["day_code"] = matches["Date"].dt.dayofweek

# Create target variable: 1 for win ('W'), 0 otherwise
matches["Target"] = (matches["Result"] == "W").astype(int)

# Split data into training and test sets based on the date
train = matches[matches["Date"] < '2024-01-01']
test = matches[matches["Date"] > '2024-01-01']

# List of predictor columns to use for the model
predictors = ["venue_code", "opp_code", "Hour", "day_code"]

# Initialize RandomForest model with specified parameters
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Train the model on the training set
rf.fit(train[predictors], train["Target"])

# Make predictions on the test set
preds = rf.predict(test[predictors])

# Calculate and print accuracy and precision scores
acc = accuracy_score(test["Target"], preds)
print("Initial accuracy score:", acc)

# Combine actual and predicted results for analysis
combined = pd.DataFrame(dict(actual=test["Target"], prediction=preds))
print("Initial precision score:", precision_score(test["Target"], preds))

# Function to calculate rolling averages for specified columns
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Columns to calculate rolling averages for
cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to each team
matches_rolling = matches.groupby("Team").apply(
    lambda x: rolling_averages(x, cols, new_cols))

matches_rolling = matches_rolling.droplevel("Team")
matches_rolling.index = range(matches_rolling.shape[0])


# Function to retrain model and make predictions on new data
def make_predictions(data, predictors):
    train = data[data["Date"] < '2024-01-01']
    test = data[data["Date"] > '2024-01-01']
    rf.fit(train[predictors], train["Target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["Target"], predicted=preds),
                            index=test.index)
    precision = precision_score(test["Target"], preds)
    return combined, precision

# Retrain model with rolling averages as additional predictors
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print("Second precision score:", precision)

# Combine home and away results for further analysis
combined = combined.merge(matches_rolling[["Date", "Team", "Opponent", "Result"]],
                          left_index=True, right_index=True)

# Dictionary to handle team name variations
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester UTD",
    "Newcastle United": "Newcastle UTD",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["Team"].map(mapping)

# Merge predictions to analyze head-to-head matchups
merged = combined.merge(combined, left_on=["Date", "new_team"],
                        right_on=["Date", "Opponent"])

# Print the count of actual wins where the model predicted a win for one team but not the other
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)][
          "actual_x"].value_counts()

# Assuming the 'Season' and 'Result' columns exist in the 'matches_rolling' DataFrame
# Predict match outcomes
matches_rolling['Predicted'] = rf.predict(matches_rolling[predictors + new_cols])

# Group by season and calculate league positions
seasons = matches_rolling.groupby("Season")

for season, data in seasons:
    # Calculate wins, losses, and draws for each team
    positions = data.groupby("Team").agg(
        wins=pd.NamedAgg(column="Predicted", aggfunc=lambda x: (x == 1).sum()),
        draws=pd.NamedAgg(column="Predicted", aggfunc=lambda x: (x == 0.5).sum()),
        losses=pd.NamedAgg(column="Predicted", aggfunc=lambda x: (x == 0).sum())
    ).sort_values("wins", ascending=False)

    # Calculate points for each team
    positions["Points"] = positions["wins"] * 3 + positions["draws"]

    # Sort the positions DataFrame by points in descending order
    sorted_positions = positions.sort_values("Points", ascending=False)

    # Print out the league table for the current season
    print(f"\nFinal Predicted Premier League Position Table for Season Season 20{season} - 20{season+1}:")
    print(sorted_positions)

    # Identifying key positions
    champions = sorted_positions.iloc[0].name
    champions_league_qualifiers = sorted_positions.iloc[:4].index.tolist()
    europa_league_qualifiers = sorted_positions.iloc[4:6].index.tolist()
    conference_league_qualifiers = sorted_positions.iloc[6:7].index.tolist()
    relegated_teams = sorted_positions.iloc[-3:].index.tolist()

    # Print summary
    print(f"\nSummary for Season 20{season} - 20{season+1}:")
    print(f"Premier League Champions: {champions}")
    print(f"Champions League Qualifiers: {', '.join(champions_league_qualifiers)}")
    print(f"Europa League Qualifiers: {', '.join(europa_league_qualifiers)}")
    print(f"Conference League Qualifiers: {', '.join(conference_league_qualifiers)}")
    print(f"Relegated Teams: {', '.join(relegated_teams)}")
