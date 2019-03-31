% Load our results
compressive_pulse_load;

% Load ABAQUS results
abaqus_displacement;
abaqus_stress;
abaqus_velocity;
abaqus_acceleration;
abaqus_external_load;

%axis_color = [138 43 226] / 255;
axis_color = [71 60 139] / 255;
axis_nohgc = [120 160 240] / 255;
%abaqus_color = [124 205 124] / 255;
abaqus_color = [188 210 238] / 255;
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
xlabel('Tempo (s)');
ylabel('Carga (N)');

% ABAQUS result
t = abaqus_load(:,1);
u = abaqus_load(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = results_compressive_load(:,1);
r = results_compressive_load(:,3);
g = plot(t,r);
set(g, 'Color', axis_color,'LineWidth',1);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot displacement at the free end
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Deslocamento axial (m)');

% ABAQUS result
t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = results_compressive_load(:,1);
u = results_compressive_load(:,2);
g = plot(t,u);
set(g, 'Color', axis_color, 'LineWidth', 1);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot stress at the 6th element (bar center approximately)
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial stress at the mid-point (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão (Pa)');

% ABAQUS result
t = abaqus_sigma(:,1);
sigma = abaqus_sigma(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = results_compressive_load(:,1);
sigma = results_compressive_load(:,6);
g = plot(t,sigma);
set(g, 'Color', axis_color,'LineWidth',1);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Velocidade axial (m/s)');

% ABAQUS result
t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = results_compressive_load(:,1);
v = results_compressive_load(:,4);
g = plot(t,v);
set(g, 'Color', axis_color,'LineWidth',1);

legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the free end
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial acceleration at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Aceleração axial (m/s^2)');

% ABAQUS result
t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = results_compressive_load(:,1);
a = results_compressive_load(:,5);
g = plot(t,a);
set(g, 'Color', axis_color,'LineWidth',1);


legend({'ABAQUS', 'Axis'}, 'Location', 'SouthEast');

