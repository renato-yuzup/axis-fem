% Load our results
rubber_results;

% Load ABAQUS results
abaqus_external_load;
abaqus_displacement_x;
abaqus_displacement_y;
abaqus_velocity_x;
abaqus_velocity_y;
abaqus_acceleration_y;
abaqus_stress_tip_xx;
abaqus_stress_tip_xy;
abaqus_stress_middle_xx;
abaqus_stress_middle_xy;

axis_color = [138 43 226] / 255;
abaqus_color = [124 205 124] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Create new figure;
f = figure;
set(f, 'Name', 'Test 3: Natural rubber beam bending test I -- Comparative results', 'Color', [1 1 1]);


%% Plot external load at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial load applied at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Load (N)');

% Our result
t = rubber_test(:,1);
r = rubber_test(:,2);
g = plot(t,r);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_load(:,1);
u = abaqus_load(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot displacement at the free end
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement-X (m)');

% Our result
t = rubber_test(:,1);
u = rubber_test(:,3);
g = plot(t,u);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_ux(:,1);
u = abaqus_ux(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity-X (m/s)');

% Our result (no hourglass control)
t = rubber_test(:,1);
v = rubber_test(:,9);
g = plot(t,v);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_vx(:,1);
v = abaqus_vx(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the free end
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Acceleration at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Acceleration-Y (m/s^2)');

% Our result (no hourglass control)
t = rubber_test(:,1);
a = rubber_test(:,12);
g = plot(t,a);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_ay(:,1);
a = abaqus_ay(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot displacement at the free end
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('Displacement at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement-Y (m)');

% Our result
t = rubber_test(:,1);
u = rubber_test(:,4);
g = plot(t,u);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_uy(:,1);
u = abaqus_uy(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,6);
hold all;
set(h, 'FontSize', 8.0);
t = title('Velocity at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity-Y (m/s)');

% Our result (no hourglass control)
t = rubber_test(:,1);
v = rubber_test(:,10);
g = plot(t,v);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_vy(:,1);
v = abaqus_vy(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');




% Create new figure;
f = figure;
set(f, 'Name', 'Test 3: Natural rubber beam bending test I -- Comparative results', 'Color', [1 1 1]);


%% Plot stress at tip element
h = subplot(2,2,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('XX stress at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,5);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_sigma_tip_xx(:,1);
sigma = abaqus_sigma_tip_xx(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

%% Plot stress at the 6th element (bar center approximately)
h = subplot(2,2,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('XX stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,7);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_sigma_mid_xx(:,1);
sigma = abaqus_sigma_mid_xx(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');




%% Plot stress at tip element
h = subplot(2,2,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('XY stress at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,6);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_sigma_tip_xy(:,1);
sigma = abaqus_sigma_tip_xy(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

%% Plot stress at the 6th element (bar center approximately)
h = subplot(2,2,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('XY stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,8);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_sigma_mid_xy(:,1);
sigma = abaqus_sigma_mid_xy(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


