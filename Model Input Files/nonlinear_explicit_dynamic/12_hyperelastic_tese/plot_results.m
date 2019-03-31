% Load our results
rubber_results;

% Load ABAQUS results
abaqus_displacement;
abaqus_velocity;
abaqus_acceleration;
abaqus_stress_edge;
abaqus_stress_4;

axis_color = [139 10 80] / 255;
abaqus_color = [188 210 238] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Create new figure;
f = figure;
set(f, 'Name', 'Test 1: Natural rubber beam linear tensile load -- Comparative results', 'Color', [1 1 1]);


%% Plot external load at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial load applied at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Carga axial (N)');

% Our result
t = rubber_test(:,1);
r = rubber_test(:,2);
g = plot(t,r);
set(g, 'Color', axis_color);

% ABAQUS result
% t = abaqus_load(:,1);
% u = abaqus_load(:,2);
% g = plot(t,u);
% set(g, 'Color', abaqus_color);

legend({'Axis'}, 'Location', 'SouthEast');


%% Plot displacement at the free end
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial displacement at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Deslocamento axial (m)');

% ABAQUS result
t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = rubber_test(:,1);
u = rubber_test(:,3);
g = plot(t,u);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot stress at tip element
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial stress at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão axial (Pa)');

% ABAQUS result
t = abaqus_sigma(:,1);
sigma = abaqus_sigma(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,6);
g = plot(t,sigma);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');



%% Plot stress at the 6th element (bar center approximately)
h = subplot(2,3,6);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão axial (Pa)');

% ABAQUS result
t = abaqus_sigma_m(:,1);
sigma = abaqus_sigma_m(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result
t = rubber_test(:,1);
sigma = rubber_test(:,7);
g = plot(t,sigma);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial velocity at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Velocidade axial (m/s)');

% ABAQUS result
t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result (no hourglass control)
t = rubber_test(:,1);
v = rubber_test(:,4);
g = plot(t,v);
set(g, 'Color', axis_color);


legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the free end
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 10.0);
t = title('Axial acceleration at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Aceleração axial (m/s^2)');

% ABAQUS result
t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color, 'LineWidth', 4);

% Our result (no hourglass control)
t = rubber_test(:,1);
a = rubber_test(:,5);
g = plot(t,a);
set(g, 'Color', axis_color);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');

