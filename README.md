# Sighthound

- plans 
   - add detailed geographic data to sighthound (soil, atmosphere, vegetation, cesium classification) and time correction


Sighthound is a python package that applies line-of-sight principles in reconstructing high resolution geolocation data from the combined output of all inter and intra vendor 
activity annotation files from consumer grade and commercially available wearable activity tracking smart watches. 
It fuses data from multiple sources, applies dynamic filtering, triangulates positions, calculates optimal and Dubin's paths, and provides structured JSON outputs. Built for researchers and analysts, it leverages mathematical models and geospatial techniques for accurate and meaningful results.

---

## Features and Theoretical Explanations

### 1. **Multi-Source GPS Data Fusion**
Combines GPS data from various formats (GPX, KML, TCX, FIT). Missing data points are handled using **linear interpolation**:

$$
x(t) = x_1 + \frac{t - t_1}{t_2 - t_1} \cdot (x_2 - x_1)
$$

Where:
- \( t \): Missing timestamp.
- $$( x(t) )$$: Interpolated value.
- $$( t_1, t_2 )$$: Known timestamps before and after \( t \).
- $$( x_1, x_2 )$$: GPS values at $$( t_1, t_2 )$$.

This ensures smooth and continuous data while maintaining temporal accuracy.

---

### 2. **Dynamic Kalman Filtering**
The Kalman filter smooths noisy GPS data and predicts missing points by modeling the system state. It has two main steps:

#### **Prediction Step**

$$ 
x_k = F \cdot x_{k-1} + w
\
\
P_k = F \cdot P_{k-1} \cdot F^T + Q
 $$

Where:
- $$( x_k )$$: State vector (e.g., position, velocity).
- ( F ): State transition matrix.
- \( w \): Process noise.
- $$( P_k )$$: Error covariance matrix.
- \( Q \): Process noise covariance.

#### **Update Step**
$$
y_k = z_k - H \cdot x_k
K_k = P_k \cdot H^T \cdot (H \cdot P_k \cdot H^T + R)^{-1}
]
[
x_k = x_k + K_k \cdot y_k
]
\
P_k = (I - K_k \cdot H) \cdot P_k
$$

Where:
- \( z_k \): Measurement vector.
- \( H \): Observation matrix.
- \( K_k \): Kalman gain.
- \( R \): Measurement noise covariance.

This filter dynamically incorporates additional metrics (e.g., speed, heart rate) to refine predictions.

---

### 3. **Triangulation**
Triangulation uses cell tower data to refine GPS positions. Weighted averaging calculates the likely position:


$$\text{Latitude}= \frac{\sum \left( \text{Latitude}_i \cdot w_i \right)}{\sum w_i}$$


$$\text{Longitude} = \frac{\sum \left( \text{Longitude}_i \cdot w_i \right)}{\sum w_i}$$


Where $$( w_i = \frac{1}{\text{Signal Strength}_i} ) for each tower ( i )$$.

This approach improves accuracy in urban or obstructed environments.

---

### 4. **Optimal Path Calculation**
The optimal path is computed using external routing APIs (e.g., Mapbox). The shortest path algorithm leverages Dijkstra’s or A*:

#### **Dijkstra’s Algorithm**

$$\text{dist}[v] = \min(\text{dist}[u] + w(u, v))$$

Where:
- $$( \text{dist}[v] ): Shortest distance to vertex ( v )$$.
- $$( w(u, v) ): Weight of the edge from ( u ) to ( v )$$.

#### **A***:
Heuristic-enhanced version of Dijkstra:
$$
f(v) = g(v) + h(v)$$
Where:
- $$( g(v) ): Cost from start to ( v )$$.
- $$( h(v) ): Heuristic estimate of cost to goal$$.

---

### 5. **Dubin's Path**
Dubin's path calculates the shortest path for vehicles or humans with turning constraints. It uses circular arcs and straight-line segments:

#### **Types of Dubin's Paths**
1. **Left-Straight-Left (LSL)**:
   $$
   \text{Path} = R \cdot (\theta_1 + \pi) + L + R \cdot (\theta_2 + \pi)
   $$
2. **Right-Straight-Right (RSR)**:
   $$
   \text{Path} = R \cdot (\theta_1) + L + R \cdot (\theta_2)
   $$

Where:$$( R )$$: Turning radius.
- $$( L ): Length of the straight segment$$.
- $$( \theta_1, \theta_2 ): Angular changes$$.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sighthound.git
cd sighthound
pip install -r requirements.txt
