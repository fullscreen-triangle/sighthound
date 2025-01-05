import numpy as np

def calculate_pdf(activity_data):
    """
    Calculate probability density function for activity data.
    Args:
        activity_data: DataFrame with 'latitude', 'longitude', and other metrics.
    Returns:
        DataFrame with PDF values.
    """
    pdf_data = activity_data.copy()
    pdf_data['probability'] = 0

    # Define Gaussian kernel parameters
    mean_lat = activity_data['latitude'].mean()
    mean_lon = activity_data['longitude'].mean()
    std_lat = activity_data['latitude'].std()
    std_lon = activity_data['longitude'].std()

    for index, row in pdf_data.iterrows():
        lat_prob = np.exp(-0.5 * ((row['latitude'] - mean_lat) / std_lat)**2)
        lon_prob = np.exp(-0.5 * ((row['longitude'] - mean_lon) / std_lon)**2)
        pdf_data.at[index, 'probability'] = lat_prob * lon_prob

    return pdf_data
