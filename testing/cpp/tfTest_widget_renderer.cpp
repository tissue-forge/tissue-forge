/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/
#include "tfTest.h"
#include "tf_system.h"
#include <iostream>

using namespace TissueForge;

// Callback function to simulate increasing noise in the simulation
void increase_noise() {
    std::cout << "Noise increased." << std::endl;
}

// Callback function to simulate decreasing noise in the simulation
void decrease_noise() {
    std::cout << "Noise decreased." << std::endl;
}

// Callback function to set the particle increment value
void set_part_incr(int val) {
    std::cout << "Particle increment set to: " << val << std::endl;
}

// Callback function to simulate adding particles to the simulation
void add_parts() {
    std::cout << "Particles added." << std::endl;
}

// Callback function to simulate removing particles from the simulation
void rem_parts() {
    std::cout << "Particles removed." << std::endl;
}

int main(int argc, char const *argv[]) {
    // Initialize the simulator configuration
    FVector3 dim(10.); // Set the dimensions of the simulation universe
    Simulator::Config config;
    config.setWindowless(true); // Run the simulation in a windowless mode
    config.universeConfig.dim = dim;
    config.universeConfig.spaceGridSize = {5, 5, 5}; // Set the grid size of the simulation space
    config.universeConfig.cutoff = 1.; // Set the cutoff distance for interactions
    TF_TEST_CHECK(tfTest_init(config)); // Initialize the simulation with the configuration

    // Create a new WidgetRenderer instance
    

    // Test setting a valid font size
    TF_TEST_CHECK(system::setWidgetFontSize(20.0f));

    // Test setting an invalid font size (should fail)
    
    TF_TEST_CHECK(system::setWidgetFontSize(50.0f));

    // Test setting a valid text color
    
    TF_TEST_CHECK(system::setWidgetTextColor("red"));

    // Test setting an invalid text color (should fail)
    TF_TEST_CHECK(system::setWidgetTextColor("invalid_color"));

    // Test setting a valid background color
    TF_TEST_CHECK(system::setWidgetBackgroundColor("blue"));

    // Test setting an invalid background color (should fail)
    TF_TEST_CHECK(system::setWidgetBackgroundColor("invalid_color"));

    // Add a button that increases the noise level when clicked
    TF_TEST_CHECK((system::addWidgetButton(increase_noise, "+Noise") != -1) ? S_OK : E_FAIL);


    // Add a button that decreases the noise level when clicked
    TF_TEST_CHECK((system::addWidgetButton(decrease_noise, "-Noise") != -1) ? S_OK : E_FAIL);

    // Add an input field for setting the particle increment value
    TF_TEST_CHECK( (system::addWidgetInputInt(set_part_incr, 1000, "Part incr.")!= -1) ? S_OK : E_FAIL);
    
    // Add an output field for displaying the current noise level
    TF_TEST_CHECK((system::addWidgetInputFloat(50.0f, "Noise") != -1)? S_OK : E_FAIL);

    // Add a button that simulates adding particles to the simulation
    TF_TEST_CHECK((system::addWidgetButton(add_parts, "+Parts") != -1)? S_OK : E_FAIL);

    // Add a button that simulates removing particles from the simulation
    TF_TEST_CHECK((system::addWidgetButton(rem_parts, "-Parts") != -1)? S_OK : E_FAIL);

    // Run the simulator for a few steps to verify the setup
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    TF_TEST_CHECK(tfTest_finalize()); // Finalize the test

    return S_OK; 
}