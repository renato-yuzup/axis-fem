% Load our results
prescribed_velocity;

% Load ABAQUS results
abaqus_displacement;
abaqus_stress;
abaqus_velocity;
abaqus_acceleration;

abaqus_displacement_midpoint;
abaqus_velocity_midpoint;
abaqus_acceleration_midpoint;

axis_color = [71 60 139] / 255;
abaqus_color = [188 210 238] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Create new figure;
f = figure;
set(f, 'Name', 'Test 7: Prescribed velocity (with Poisson) -- Comparative results', 'Color', [1 1 1]);



%% 1) Plot nodal velocity at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial velocity (m/s)');

% ABAQUS result
t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
v = results(:,3);
g = plot(t,v);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');



%% 2) Plot nodal velocity at the mid-nodes
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial velocity (m/s)');

% ABAQUS result
t = abaqus_v_midpoint(:,1);
v = abaqus_v_midpoint(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
v = results(:,6);
g = plot(t,v);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');



%% 3) Plot displacement at the free end
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

% ABAQUS result
t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
u = results(:,2);
g = plot(t,u);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');



%% 4) Plot displacement at the mid-nodes
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

% ABAQUS result
t = abaqus_u_midpoint(:,1);
u = abaqus_u_midpoint(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
u = results(:,5);
g = plot(t,u);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');



%% 5) Plot nodal acceleration at the free end
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial acceleration at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial acceleration (m/s^2)');

% ABAQUS result
t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
a = results(:,4);
g = plot(t,a);
set(g, 'Color', axis_color);

% Since little difference is expected in results, freeze
% y-axis limits in order to prevent strange MATLAB behavior
ax = axis;
max_a = max(abaqus_a(:,2));
ax(3:4) = [-2*max_a 2*max_a];
axis(ax);


legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% 6) Plot nodal acceleration at the mid-nodes
h = subplot(2,3,6);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial acceleration at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial acceleration (m/s^2)');

% ABAQUS result
t = abaqus_a_midpoint(:,1);
a = abaqus_a_midpoint(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
a = results(:,7);
g = plot(t,a);
set(g, 'Color', axis_color);



legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');

figure;

%% 8) Plot stress at the 6th element (bar center approximately)
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% ABAQUS result
t = abaqus_sigma(:,1);
sigma = abaqus_sigma(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = results(:,1);
sigma = results(:,8);
g = plot(t,sigma);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'NorthEast');
