import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, columns=None):
    """Plot distributions of key numerical columns."""
    if columns is None:
        columns = ['TotalPremium', 'TotalClaims', 'LossRatio']
    
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))
    for ax, col in zip(axes, columns):
        sns.histplot(df[col], ax=ax, kde=True)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    return fig

def plot_correlations(df, columns=None):
    """Plot correlation matrix for numerical columns."""
    if columns is None:
        columns = ['TotalPremium', 'TotalClaims', 'LossRatio']
    
    corr = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    return plt.gcf()