/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 
 *  Updated By: Stephen Giardinelli
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen_;

void ParticleFilter::init(double x, double y, double theta, double stdev[]) {

	//Set number of particles to be used
	num_particles_ = 200;

	// Create gaussian distributions for x, y, and theta
	normal_distribution<double> x_distribution(0, stdev[0]);
	normal_distribution<double> y_distribution(0, stdev[1]);
	normal_distribution<double> theta_distribution(0, stdev[2]);

	// Initialize all N particles
	for (int i = 0; i < num_particles_; ++i){
		// Instantiate particle
		Particle particle;
		
		// Set particle ID 
		particle.id = i;

		// Set x, y, and theta  and add noise
		particle.x = x + x_distribution(gen_);
		particle.y = y + y_distribution(gen_);
		particle.theta = theta + theta_distribution(gen_);

		// Set particle weight to default of 1
		particle.weight = 1.0;

		// Push particle into vector
		particles_.push_back(particle);
	}

	// Update initialization flag
	is_initialized_ = true;
	return;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// Create gaussian distributions for x, y, and theta
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	for(auto&& particle : particles_){
		// If the yaw_rate = 0 update x and y with motion and noise
		if ( fabs(yaw_rate) < 0.00001) {
			particle.x += velocity * delta_t * cos( particle.theta ) + x_noise(gen_);
			particle.y += velocity * delta_t * sin( particle.theta ) + y_noise(gen_);
		} 
		else { // If the yaw rate != 0, update x, y and theta with motion and noise
			particle.x += velocity / yaw_rate * ( sin( particle.theta + yaw_rate * delta_t ) 
									- sin(particle.theta) ) + x_noise(gen_);
			
			particle.y += velocity / yaw_rate * ( cos(particle.theta) 
									- cos(particle.theta + yaw_rate * delta_t) ) + y_noise(gen_);
			
			particle.theta += yaw_rate * delta_t + theta_noise(gen_);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	// Iterate through each landmark observation
	for(auto&& obs : observations){

		// Initialize minimum distance and landmark ID variables
		double min_distance = 99999999999.0;
		int landmark_id = -1;

		// Iterate through each predicted landmark
		for(auto&& pred : predicted){

			// Calculate the euclidean distance between the observed and predicted landmarks
			double euc_dist = dist(obs.x, obs.y, pred.x, pred.y);

			// If this is the smallest distance, save it alone with the landmark ID
			if(euc_dist < min_distance){
				min_distance = euc_dist;
				landmark_id = pred.id;
			}
		}
		// Update the landmark ID in observations
		obs.id = landmark_id;
	}
	return;


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	double stdx = std_landmark[0];
	double stdy = std_landmark[1];

	// Iterate through vector of particles
	for(auto&& particle : particles_){

		// Create vector of LandmarkObs
		vector<LandmarkObs> landmarks_in_range;

		//Extract Values from particle
		double x = particle.x;
		double y = particle.y;
		double theta = particle.theta;

		// Iterate through each landmark in the map's landmark list
		for(auto&& map_landmark : map_landmarks.landmark_list){

			// If the landmark is within sensor range, push it ot the back of the landmark_preds vector
			if(dist(x, y, map_landmark.x_f, map_landmark.y_f) <= sensor_range){
				landmarks_in_range.push_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
			}
		}

		// Create vector to hold transformed landmark observations
		vector<LandmarkObs> observations_trans;
		for(auto&& obs : observations){
			double transformed_x = cos(theta) * obs.x - sin(theta) * obs.y + particle.x;
			double transformed_y = sin(theta) * obs.x + cos(particle.theta) * obs.y + particle.y;
			observations_trans.push_back(LandmarkObs{obs.id, transformed_x, transformed_y});
		}

		// Perform the data association
		dataAssociation(landmarks_in_range, observations_trans);

		// Reset the weight to 1.0
		particle.weight = 1.0;

		// Loop to calculate weights
		for(auto&& obs_trans : observations_trans){
			int associated_id = obs_trans.id;
			double pred_x, pred_y;

			// Iterate through all landmarks in range to find matching
			// lanmark ID.  Once found, break for loop for efficiency
			for(auto&& ldmk_in_range : landmarks_in_range){
				if(ldmk_in_range.id == associated_id){
					pred_x = ldmk_in_range.x;
					pred_y = ldmk_in_range.y;
					break;
				}
			}

			// Calculate the weight value based on location
			double std_x_2 = stdx*stdx;
			double std_y_2 = stdy*stdy;
			double dx_2 = (obs_trans.x - pred_x)*(obs_trans.x - pred_x);
			double dy_2 = (obs_trans.y - pred_y)*(obs_trans.y - pred_y);

			double obs_weight = ( 1 / (2*M_PI*stdx*stdy) ) * exp( -( dx_2 / (2 * std_x_2 )
										+ ( dy_2 / (2 * std_y_2 ) ) ) );

			// Multiply this into the existing weight
			particle.weight *= obs_weight;
		}
	}
	return;
	
}

void ParticleFilter::resample() {

	// Create vector for all of the weights
	vector<double> weights;
	for(auto particle : particles_){
		weights.push_back(particle.weight);
	}
	// Generate starting index for resample
	uniform_int_distribution<int> int_dist(0, num_particles_ - 1);

	// Set index from distribution
	int index = int_dist(gen_);

	// Extract the maximum weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// Create a uniform random distribution
	uniform_real_distribution<double> distribution(0.0, max_weight);

	// Initialize beta to 0.0
	double beta = 0.0;

	// Create vector for new particles
	vector<Particle> resampled_particles;

	// Wheel method
	for(int i = 0; i < num_particles_; ++i){
		beta += 2.0 * distribution(gen_);
		while(beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles_;
		}
		resampled_particles.push_back(particles_[index]);
	}

	// Update particles_ vector with resampled particles
	particles_ = resampled_particles;
	return;
	
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
