#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;

  // state dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  double w0 = lambda_ / (lambda_ + n_aug_);
  double w = 1 / (2 * (lambda_ + n_aug_));
  weights_.fill(w);
  weights_(0) = w0;

  // Time when the state is true, in us
  time_us_ = 0;

  // Set NIS
  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << pow(std_laspx_, 2), 0,
              0, pow(std_laspy_, 2);

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radr_, 2), 0, 0,
              0, pow(std_radphi_, 2), 0,
              0, 0, pow(std_radrd_, 2);

  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if (!is_initialized_)
   {
       if (meas_package.sensor_type_ == MeasurementPackage::LASER)
       {
           // Initialize the state vector
           x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0,0.0;

           // Initialize the state covariance matrix
           P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
                 0, std_laspy_ * std_laspy_, 0, 0, 0,
                 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1;
       }
       else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
       {
           double rho = meas_package.raw_measurements_[0];
           double phi = meas_package.raw_measurements_[1];
           double rhod = meas_package.raw_measurements_[2];

           // calculate position
           double px = rho * cos(phi);
           double py = rho * sin(phi);

           // calculate velocity
           double vx = rhod * cos(phi);
           double vy = rhod * sin(phi);
           double v = sqrt(vx * vx + vy * vy);

           // Initialize the state vector
           v_ << px, py, v, rho, rhod;

           // Initialize the state covariance matrix
           P_ << 1, 0, 0, 0, 0,
                 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 1;
       }
       time_us_ = meas_package.timestamp_;
       is_initialized_ = true;
   }
   else
   {
       double dt = (meas_package.timestamp_ - time_us_) / 1e6;
       time_us_ = meas_package.timestamp_;

       Prediction(dt);

       if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
           UpdateRadar(meas_package);
       else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
           UpdateLidar(meas_package);
   }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
   MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
   GenerateAugmentedSigmaPoints(&Xsig_aug);
   SigmaPointPrediction(Xsig_aug, &Xsig_pred_, delta_t);
   PredictMeanAndCovariance(&x_, &P_);
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* Pout)
{
    // create vector for predicted state
    VectorXd x = VectorXd(n_x);

    // create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x, n_x);

    // set weights
    weights.fill(0.5/(n_aug+lambda));
    weights(0) = lambda / (lambda + n_aug);

    // predict state mean
    x = Xsig_pred * weights;

    // predict state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug + 1; ++i)
    {  // iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose() ;
    }

    // write result
    *x_out = x;
    *P_out = P;
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd& Xsig_out)
{
    // create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    // create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

    // create augmented mean state
    x_aug.fill(0.0);
    x_aug(n_x_) = x_;

    // create augmented covariance matrix
    P_aug.fill(0);
    P_aug.topLeftCorner(5,5) = P;
    P_aug(5,5) = std_a*std_a;
    P_aug(6,6) = std_yawdd*std_yawdd;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug; ++i)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda+n_aug) * L.col(i);
        Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda+n_aug) * L.col(i);
    }

    *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd& x_pred, MatrixXd* Xsig_out, double delta_t)
{
    // Create matrix with predicted sigma points as columns
    MatirxXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // predicted sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        // extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        // write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }
    *Xsig_out = Xsig_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
   VectorXd z = meas_package.raw_measurements_;
   VectorXd z_pred = H_laser_ * x;
   VectorXd y = z - z_pred;

   MatrixXd Ht = H_laser_.transpose();
   MatrixXd S = H_laser_ * P_ * ht + R_laser_;
   MatrixXd Sinv = S.inverse();
   MatrixXd PHt = P_ * Ht;
   MatrixXd K = PHt * Sinv;
   MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

   // update state mean and covariance matrix
   x_ = x_ + (K * y);
   P_ = (I - K * H_laser_) * P_;

   NIS_laser_ = y.transpose() * Sinv * y;
   std::ofstream NIS_laser_file_;
   if (!NIS_laser_file_.is_open())
       NIS_laser_file_.open("../res/NIS_Lidar.csv", std::fstream::out | std::fstream::app);
   NIS_laser_file_ << NIS_laser_ << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   /* predict radar measurement */
    // set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {  // 2n+1 simga points
        // extract values for better readability
        double p_x = Xsig_pred(0,i);
        double p_y = Xsig_pred(1,i);
        double v  = Xsig_pred(2,i);
        double yaw = Xsig_pred(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
        Zsig(1,i) = atan2(p_y,p_x);                                // phi
        Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
    }

    // mean predicted measurement
    z_pred.fill(0.0);
    z_pred = Zsig * weights;

    // innovation covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    S = S + R_radar_;

    /* update */
    // create example vector for incoming radar measurement
    VectorXd z = meas_package.raw_measurements_;

    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    // calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    std::ofstream NIS_radar_file_;
    if (!NIS_radar_file_.is_open())
        NIS_radar_file_.open("../res/NIS_Radar.csv", std::fstream::out | std::fstream::app);
    NIS_radar_file_ << NIS_radar_ << std::endl;
}