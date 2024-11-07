import csv

input_file = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.diag'
output_file = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'

# Open the input .diag file in read mode and the output CSV file in write mode
with open(input_file, 'r') as diag_file, open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Add header w/ column names
    header = ['Timestep', 'Particle Number', 'Mass', 'X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity', 'Phase']
    csv_writer.writerow(header)
    
    lines = diag_file.readlines()
    i = 0
    total_timesteps = 0
    
    # First pass to count the number of timesteps
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Read the first line of each block which contains the timestep and number of bodies
        timestep_data = lines[i].strip().split()
        
        # Ensure the line contains at least two elements (timestep and number of bodies)
        if len(timestep_data) < 2:
            i += 1
            continue
        
        total_timesteps += 1  # Count this timestep block
        num_bodies = int(timestep_data[1])  # Extract number of particles
        i += num_bodies + 1  # Skip the lines with particle data and the timestep line
    
    # Calculate the midpoint for phase assignment
    midpoint = total_timesteps // 2
    print(f"Total timesteps: {total_timesteps}, Midpoint: {midpoint}")
    
    # Reset index to process the file again
    i = 0
    current_timestep_index = 0  # Initialize the index for phases
    
    # Second pass to read and write data with correct phase
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Read the first line of each block which contains the timestep and number of bodies
        timestep_data = lines[i].strip().split()
        
        # Ensure the line contains at least two elements (timestep and number of bodies)
        if len(timestep_data) < 2:
            i += 1
            continue
        
        timestep = float(timestep_data[0])  # Extract timestep
        num_bodies = int(timestep_data[1])  # Extract number of particles
        
        # Determine the phase based on current timestep index
        if current_timestep_index < midpoint:
            phase = 1
        elif current_timestep_index == midpoint:
            phase = 0
        else:
            phase = -1
        
        # Read data for each particle
        for particle_num in range(num_bodies):
            if i >= len(lines):
                print("Reached end of file unexpectedly")
                break
            
            particle_data = lines[i + particle_num + 1].strip().split()  # Read the particle data line
            
            # Ensure the line has exactly 7 values (mass, x/y/z positions, x/y/z velocities)
            if len(particle_data) != 7:
                print(f"Skipping line {i + particle_num + 1}: {lines[i + particle_num + 1].strip()} - invalid particle data")
                continue
            
            mass = particle_data[0]
            x_pos = particle_data[1]
            y_pos = particle_data[2]
            z_pos = particle_data[3]
            x_vel = particle_data[4]
            y_vel = particle_data[5]
            z_vel = particle_data[6]
            
            # Write particle data to CSV, including the phase
            csv_writer.writerow([timestep, particle_num + 1, mass, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, phase])
        
        # Update the index and increase the current timestep index
        i += num_bodies + 1  # Move to the next timestep block
        current_timestep_index += 1  # Increment the timestep index

print("Conversion complete.")
