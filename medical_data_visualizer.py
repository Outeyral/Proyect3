import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
df = pd.read_csv('medical_examination.csv')

# Calculate BMI and determine if person is overweight
df['overweight'] = (df['weight'] / (df['height'] / 100)**2) > 25
df['overweight'] = df['overweight'].astype(int)

# Normlize cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Create Categoric Plot
def draw_cat_plot():
    # Transform data in long format
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group by cardio, variable and value
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Create Cat Plot with Seaborn
    fil = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    # Variable fig
    fig = fil.fig


    # Download and return figure
    fig.savefig('catplot.png')
    return fig


# Create Heat Map
def draw_heat_map():
    # Clean data with logical conditions
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Create mask for upper half of heat map
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Config Matplotlib fig
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw heat map with Seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', center=0)

    # Download and return fig
    fig.savefig('heatmap.png')
    return fig
