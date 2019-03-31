% Load our results
plasticity_results;

% Load ABAQUS results
abaqus_external_load;
abaqus_displacement;
abaqus_velocity;
abaqus_acceleration_bottom;
abaqus_stress_xx;
abaqus_stress_yy;
abaqus_stress_xy;
abaqus_plastic_strain_yy;
abaqus_plastic_strain_xy;
abaqus_equiv_plastic_strain;

axis_color = [139 10 80] / 255;
abaqus_color = [188 210 238] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

axis_t    = plastic_test(:,1);
axis_f    = plastic_test(:,2);
axis_uy   = plastic_test(:,3);
axis_sxx  = plastic_test(:,4);
axis_syy  = plastic_test(:,5);
axis_sxy  = plastic_test(:,6);
axis_vy   = plastic_test(:,7);
axis_ay   = plastic_test(:,8);
axis_ep   = plastic_test(:,9);
axis_psxx = plastic_test(:,10);
axis_psyy = plastic_test(:,11);
axis_pszz = plastic_test(:,12);
axis_psyz = plastic_test(:,13);
axis_psxz = plastic_test(:,14);
axis_psxy = plastic_test(:,15);

% Create new figure;
f = figure;
set(f, 'Name', 'Test 6: Bilinear plasticity test I -- Comparative results', 'Color', [1 1 1]);


%% Plot external load at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 10.0);
t = title('Transversal load applied at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Carga transversal (N)');

% ABAQUS result
t = abaqus_load(:,1);
u = abaqus_load(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
r = axis_f;
g = plot(t,r);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot displacement at the free end
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 10.0);
t = title('Transversal displacement at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Deslocamento transversal (m)');

% ABAQUS result
t = abaqus_uy(:,1);
u = abaqus_uy(:,2);
g = plot(t,u);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
u = axis_uy;
g = plot(t,u);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot nodal velocity at the free end
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 10.0);
t = title('Transversal velocity at the free end');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Velocidade transversal (m/s)');

% ABAQUS result
t = abaqus_vy(:,1);
v = abaqus_vy(:,2);
g = plot(t,v);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = axis_t;
v = axis_vy;
g = plot(t,v);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot nodal acceleration at the free end
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 10.0);
t = title('Transversal acceleration at the free end (bottom only)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Aceleração transversal (m/s^2)');

% ABAQUS result
t = abaqus_ay_bottom(:,1);
a = abaqus_ay_bottom(:,2);
g = plot(t,a);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result (no hourglass control)
t = axis_t;
a = axis_ay;
g = plot(t,a);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');





% Create new figure;
f = figure;
set(f, 'Name', 'Test 6: Bilinear plasticity test I -- Comparative results', 'Color', [1 1 1]);

%% Plot xx-stress at tip element
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 10.0);
t = title('XX stress at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão XX (Pa)');

% ABAQUS result
t = abaqus_sigma_xx(:,1);
sigma = abaqus_sigma_xx(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
sigma = axis_sxx;
g = plot(t,sigma);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot yy-stress at tip element
% h = subplot(2,2,2);
% hold all;
% set(h, 'FontSize', 10.0);
% t = title('YY stress at tip element');
% set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
% set(t, 'BackgroundColor', txtBgColor);
% xlabel('Tempo (s)');
% ylabel('Tensão YY (Pa)');

% % ABAQUS result
% t = abaqus_sigma_yy(:,1);
% sigma = abaqus_sigma_yy(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color,'LineWidth',4);

% % Our result
% t = axis_t;
% sigma = axis_syy;
% g = plot(t,sigma);
% set(g, 'Color', axis_color);

% legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot xy-stress at tip element
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('XY stress at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Tensão XY (Pa)');

% ABAQUS result
t = abaqus_sigma_xy(:,1);
sigma = abaqus_sigma_xy(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
sigma = axis_sxy;
g = plot(t,sigma);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot equivalent plastic strain at tip element
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 10.0);
t = title('Equivalent plastic strain at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Deformação plástica equivalente');

% ABAQUS result
t = abaqus_epstrain(:,1);
ep = abaqus_epstrain(:,2);
g = plot(t,ep);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
ep = axis_ep;
g = plot(t,ep);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot yy-plastic strain at tip element
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 10.0);
t = title('YY plastic strain at tip element');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Tempo (s)');
ylabel('Deformação plástica YY');

% ABAQUS result
t = abaqus_pstrain_yy(:,1);
ps = abaqus_pstrain_yy(:,2);
g = plot(t,ps);
set(g, 'Color', abaqus_color,'LineWidth',4);

% Our result
t = axis_t;
ps = axis_psyy;
g = plot(t,ps);
set(g, 'Color', axis_color);

legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');


%% Plot xy-plastic strain at tip element
% h = subplot(2,2,6);
% hold all;
% set(h, 'FontSize', 10.0);
% t = title('XY plastic strain at tip element');
% set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
% set(t, 'BackgroundColor', txtBgColor);
% xlabel('Tempo (s)');
% ylabel('Deformação plástica XY');

% % ABAQUS result
% t = abaqus_pstrain_xy(:,1);
% ps = abaqus_pstrain_xy(:,2);
% g = plot(t,ps);
% set(g, 'Color', abaqus_color,'LineWidth',4);

% % Our result
% t = axis_t;
% ps = axis_psxy;
% g = plot(t,ps);
% set(g, 'Color', axis_color);

% legend({'ABAQUS','Axis'}, 'Location', 'SouthEast');

