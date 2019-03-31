% Load our results
cook_exercise;

% Load ABAQUS results
abaqus_displacement;
abaqus_stress;
abaqus_internal_force;
abaqus_velocity;
abaqus_acceleration;
abaqus_reaction;
abaqus_displacement_midnodes;
abaqus_velocity_midnodes;
abaqus_acceleration_midnodes;

axis_color = [138 43 226] / 255;
abaqus_color = [124 205 124] / 255;
exact_color = [0 191 255] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Create new figure;
f = figure;
set(f, 'Name', 'Test 1: Cook example -- Comparative results', 'Color', [1 1 1]);

%% Plot displacement at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (in)');

% Our result
t = results(:,1);
u = results(:,2);
g = plot(t,u);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'NorthEast');

%% Plot stress at element 20 (bar center)
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial stress at the mid-point (element 20)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão (psi)');

% Our result
t = results(:,1);
sigma = results(:,3);
ss = smooth(sigma, 25);
g = plot(t,sigma);
set(g, 'Color', axis_color);

% Exact result
tempo = 1e-3 * [0 0.05 0.0500000001 0.15 0.1500000001 0.2];
sigma_exato = [0 0 -100 -100 -200 -200];
g = plot(tempo, sigma_exato, '--');
set(g, 'Color', exact_color);

[xData, yData] = prepareCurveData( t, sigma );

% Set up fittype and options.
ft = fittype( 'gauss8' );
opts = fitoptions( ft );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0];
opts.StartPoint = [0 -1.3371263679766 0.382036057019173 0 -0.955090310957426 0.382036057019173 0 -0.573054253938253 0.382036057019173 0 -0.191018196919081 0.382036057019173 0 0.191017860100092 0.382036057019173 0 0.573053917119265 0.382036057019173 0 0.955089974138438 0.382036057019173 0 1.33712603115761 0.382036057019173];
opts.Upper = [Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf];
opts.Normalize = 'on';

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

g = plot( fitresult );

legend({'Axis', 'Exato', 'Smooth'}, 'Location', 'NorthEast');

%% Plot internal force at the free end
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('Internal force at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial force (lb)');

% Our result
t = results(:,1);
fint = results(:,4);
g = plot(t,fint);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_fint(:,1);
fint = abaqus_fint(:,2);
g = plot(t,fint);
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
ylabel('Axial velocity (in/s)');

% Our result
t = results(:,1);
v = results(:,5);
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
ylabel('Axial acceleration (in/s^2)');

% Our result
t = results(:,1);
a = results(:,6);
g = plot(t,a);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot reaction force acting in the center node at the fixed end
h = subplot(2,3,6);
hold all;
set(h, 'FontSize', 8.0);
t = title('Reaction force in the center node at the fixed end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial force (lb)');

% Our result
t = results(:,1);
r = -results(:,7);
g = plot(t,r);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_r(:,1);
r = abaqus_r(:,2);
g = plot(t,r);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');





%%
figure;

%% Plot displacement at the mid-nodes
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial displacement at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (in)');

% Our result
t = results(:,1);
u = results(:,8);
g = plot(t,u);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_u_midnodes(:,1);
u = abaqus_u_midnodes(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'NorthEast');


%% Plot nodal velocity at the mid-nodes
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial velocity at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial velocity (in/s)');

% Our result
t = results(:,1);
v = results(:,9);
g = plot(t,v);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_v_midnodes(:,1);
v = abaqus_v_midnodes(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the mid-nodes
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('Axial acceleration at the mid-nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Axial acceleration (in/s^2)');

% Our result
t = results(:,1);
a = results(:,10);
g = plot(t,a);
set(g, 'Color', axis_color);

% ABAQUS result
t = abaqus_a_midnodes(:,1);
a = abaqus_a_midnodes(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


