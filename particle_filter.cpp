
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <sstream>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	 /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  // Lesson 5 section 6 shows example of codes for i initialization by Gaussian sampling
	num_particles = 40;
	default_random_engine gen;
	// defining normal distributions for sensor noise:
    normal_distribution<double> d_theta(theta, std[2]);
    normal_distribution<double> d_x(x, std[0]);
	normal_distribution<double> d_y(y, std[1]);
	int i;
	for (i = 0; i < num_particles; i++) {
	  Particle c_particle;
      c_particle.theta = d_theta(gen);   
	  c_particle.x = d_x(gen);
	  c_particle.y = d_y(gen);
      c_particle.id = i;
	  c_particle.weight = 1.0;
	  
	  particles.push_back(c_particle);
	  weights.push_back(c_particle.weight);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	/**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
// from concepts in lesosn 5, section 9
	default_random_engine gen;
	
	int i;
	for (i = 0; i < num_particles; i++) {
	  double p_x, p_y, p_theta;
	  /*For calculate new state, Decided to consider Checking against low values of yaw_rate instead of       checking against zero. This gives
        more flexibity and optimization and is closer to reality in real world*/
	  if (abs(yaw_rate) < 0.001) {
	    p_theta = particles[i].theta;
        p_x = particles[i].x + delta_t*velocity * cos(particles[i].theta) ;  //particle location in global map coordinate system
	    p_y = particles[i].y + delta_t*velocity * sin(particles[i].theta) ;  //particle location in global map coordinate system
	    
	  } else{
        
	    p_theta = particles[i].theta + (delta_t * yaw_rate);
        p_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (delta_t * yaw_rate)) - sin(particles[i].theta));
	    p_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta+ (delta_t * yaw_rate)));
	  }
	  //creating gaussian distributions
	  normal_distribution<double> d_theta(p_theta, std_pos[2]);
      normal_distribution<double> d_x(p_x, std_pos[0]);
	  normal_distribution<double> d_y(p_y, std_pos[1]);
	  // considering noise:
	  particles[i].x = d_x(gen);
	  particles[i].y = d_y(gen);
	  particles[i].theta = d_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	/**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	
  // using lesson 5, section 11 learnings
	for (unsigned int n = 0; n < observations.size(); n++) {
		double observe_x = observations[n].x;
		double observe_y = observations[n].y;
        double l_dist =  sqrt(2)*sensor_range;
		int n_l_id = -1;  // nearest landmark
		
		for (unsigned int m = 0; m < predicted.size(); m++) {
		  int p_id = predicted[m].id;
          double p_x = predicted[m].x;
		  double p_y = predicted[m].y;
          // getting distance between current/predicted landmarks:
		  double c_dist = dist(observe_x, observe_y, p_x, p_y);

		  if (c_dist < l_dist) {
		    l_dist = c_dist;
		    n_l_id = p_id;
		  }
		}
       // setting the observation's id to the closest predicted landmark's id:
		observations[n].id = n_l_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
	/**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    double m_x, m_y, c_x, c_y, exp1, mu_x, mu_y, sigmax, sigmay, gaussian_n, gaussian_d_x, gaussian_d_y, p_x, p_y, p_theta,          weight, distance_l, mark_observation;
  //m_x, m_y: transformed observation
  // c_x, c_y: observations in vehicle's coordinate system
  //p_x, p_y: location of particle in global map coordinate system
    vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
    sigmax = std_landmark[0];
    sigmay = std_landmark[1];
    gaussian_d_x = 2 * sigmax * sigmax;
    gaussian_d_y = 2 *sigmay * sigmay;
    gaussian_n = pow(2 * M_PI * sigmax * sigmay,-1);
    for(int l=0; l<num_particles; ++l){
        weight = 1.0;
        p_x = particles[l].x;
        p_y = particles[l].y;
        p_theta = particles[l].theta;
        vector<LandmarkObs> closest_l;
        LandmarkObs closest_l_p;
      //finding closest landmarks
        for(unsigned int m=0; m<landmarks.size(); ++m){
            distance_l = dist(p_x, p_y, landmarks[m].x_f, landmarks[m].y_f);
          //picking up desired ;landmarks within range
            if(distance_l < sensor_range){
                closest_l_p.x = landmarks[m].x_f;
                closest_l_p.y = landmarks[m].y_f;
                closest_l_p.id = landmarks[m].id_i;
                closest_l.push_back(closest_l_p);
            }  }
        for(unsigned int n=0; n<observations.size(); ++n){
            c_x = observations[n].x;
            c_y = observations[n].y;
            m_x = p_x + cos(p_theta)*c_x - sin(p_theta)*c_y;
            m_y = p_y + sin(p_theta)*c_x + cos(p_theta)*c_y;

            double min_observation = 100;
            int min_index = 0;
          //finding closest index
            for(unsigned int o=0; o<closest_l.size(); ++o){
                mark_observation = dist(m_x, m_y, closest_l[o].x, closest_l[o].y);
                if(mark_observation < min_observation){
                    min_index = closest_l[o].id;
                    min_observation = mark_observation;
                } }
          //finding closest position
            for(unsigned int p=0; p<closest_l.size(); ++p){
                if(closest_l[p].id == min_index){
                    mu_x = closest_l[p].x;
                    mu_y = closest_l[p].y;
                    break;
                } }
            exp1 = (m_x-mu_x)*(m_x-mu_x)/gaussian_d_x + (m_y-mu_y)*(m_y-mu_y)/gaussian_d_y;
            weight *= gaussian_n * exp(-exp1);
        }
        particles[l].weight = weight;
        weights[l] = weight;
    }}

void ParticleFilter::resample() {
	/**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    
    int index = rand() % num_particles;
    double max_weight = *max_element(weights.begin(), weights.end());
    vector<Particle> p_new (num_particles);
    double betta = 0;
    
    for(int i=0; i<num_particles; ++i){

        betta = betta+ (rand() / (RAND_MAX + 1.0)) * (2*max_weight);
        while(weights[index]<betta){
            betta = betta - weights[index];
            index = (index+1) % num_particles;
        }
        p_new[i] = particles[index];
    }
    particles = p_new;


    // cout << "debug state: resampled completed." << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	 // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

	
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	particle.associations= associations;
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
    s = s.substr(0, s.length()-1); 
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  
    return s;
}