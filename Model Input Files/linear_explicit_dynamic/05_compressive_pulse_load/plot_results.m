% Load our results
compressive_pulse_load;

% Load ABAQUS results
abaqus_displacement;
abaqus_stress;
abaqus_velocity;
abaqus_acceleration;
abaqus_external_load;

axis_color = [138 43 226] / 255;
axis_nohgc = [120 160 240] / 255;
abaqus_color = [124 205 124] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Create new figure;
f = figure;
set(f, 'Name', 'Test 5: Compressive pulse load (with Poisson) -- Comparative results', 'Color', [1 1 1]);


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
t = results_compressive_load(:,1);
r = results_compressive_load(:,3);
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
t = title('Axial displacement at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

% Our result (no hourglass control)
t = results_compressive_load(:,1);
u = results_compressive_load(:,2);
g = plot(t,u);
set(g, 'Color', axis_color,'Marker','o');

% ABAQUS result
t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color,'Marker','+');

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot stress at the 6th element (bar center approximately)
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result (no hourglass control)
t = results_compressive_load(:,1);
sigma = results_compressive_load(:,6);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_sigma(:,1);
sigma = abaqus_sigma(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial velocity (m/s)');

% Our result (no hourglass control)
t = results_compressive_load(:,1);
v = results_compressive_load(:,4);
g = plot(t,v);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the free end
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial acceleration at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial acceleration (m/s^2)');

% Our result (no hourglass control)
t = results_compressive_load(:,1);
a = results_compressive_load(:,5);
g = plot(t,a);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_a(:,1);
a = smooth(abaqus_a(:,2));
g = plot(t,a);
set(g, 'Color', abaqus_color);


legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

