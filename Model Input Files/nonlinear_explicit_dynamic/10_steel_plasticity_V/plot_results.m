% Load our results
plasticity_results;
L = 0.1;

axis_color = [138 43 226] / 255;
abaqus_color = [124 205 124] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

color_eng     = [255 127 0] / 255;
color_log     = [160 82 45] / 255;
color_green   = [127 255 0] / 255;
color_almansi = [100 149 237] / 255;

axis_t    = plastic_test(:, 1);
axis_uy   = plastic_test(:, 2);
axis_sxx  = plastic_test(:, 3);
axis_syy  = plastic_test(:, 4);
axis_szz  = plastic_test(:, 5);
axis_syz  = plastic_test(:, 6);
axis_sxz  = plastic_test(:, 7);
axis_sxy  = plastic_test(:, 8);
axis_vy   = plastic_test(:, 9);
axis_ay   = plastic_test(:,10);
axis_ep   = plastic_test(:,11);
axis_psxx = plastic_test(:,12);
axis_psyy = plastic_test(:,13);
axis_pszz = plastic_test(:,14);
axis_psyz = plastic_test(:,15);
axis_psxz = plastic_test(:,16);
axis_psxy = plastic_test(:,17);
axis_exx  = plastic_test(:,18);
axis_eyy  = plastic_test(:,19);
axis_ezz  = plastic_test(:,20);
axis_eyz  = plastic_test(:,21);
axis_exz  = plastic_test(:,22);
axis_exy  = plastic_test(:,23);

% Create new figure;
f = figure;
set(f, 'Name', 'Test 10: Bilinear plasticity test IV -- Comparative results', 'Color', [1 1 1]);

%% Plot displacement at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Displacement at the top');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement-Y (m)');

% Our result
t = axis_t;
u = axis_uy;
g = plot(t,u);
set(g, 'Color', axis_color);




%% Plot xx-stress
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('XX stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_sxx;
g = plot(t,sigma);
set(g, 'Color', axis_color);


%% Plot yy-stress
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('YY stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_syy;
g = plot(t,sigma);
set(g, 'Color', axis_color);


%% Plot zz-stress
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('ZZ stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_szz;
g = plot(t,sigma);
set(g, 'Color', axis_color);




%% Plot xz-stress
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('XZ stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_sxz;
g = plot(t,sigma);
set(g, 'Color', axis_color);



%% Plot XY-stress
h = subplot(2,3,6);
hold all;
set(h, 'FontSize', 8.0);
t = title('XY stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_sxy;
g = plot(t,sigma);
set(g, 'Color', axis_color);









% Create new figure;
f = figure;
set(f, 'Name', 'Test 10: Bilinear plasticity test IV -- Comparative results', 'Color', [1 1 1]);


%% Plot equivalent plastic strain
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Equivalent plastic strain');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Equivalent plastic strain');

% Our result
t = axis_t;
ep = axis_ep;
g = plot(t,ep);
set(g, 'Color', axis_color, 'LineWidth', 2.0);

e = axis_uy / L; % Engineering strain
elog = log(1 + e);
lambda = 1 + e;
eg = 0.5 * (lambda.^2 - 1);
ee = 0.5 * (1 - 1 / lambda.^2);

g = plot(t,e);
set(g, 'Color', color_eng);
g = plot(t,elog);
set(g, 'Color', color_log);
g = plot(t,eg);
set(g, 'Color', color_green);
g = plot(t,ee);
set(g, 'Color', color_almansi);

legend({'Axis (unknown measure)', 'Engineering', 'Logarithmic', 'Green-Lagrange', 'Almansi'}, 'Location', 'SouthEast');







%% Plot xx-plastic strain at tip element
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('XX plastic strain');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('XX-Plastic strain');

% Our result
t = axis_t;
ps = axis_psxx;
g = plot(t,ps);
set(g, 'Color', axis_color);



%% Plot yy-plastic strain at tip element
h = subplot(2,3,3);
hold all;
set(h, 'FontSize', 8.0);
t = title('YY plastic strain');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('YY-Plastic strain');

% Our result
t = axis_t;
ps = axis_psyy;
g = plot(t,ps);
set(g, 'Color', axis_color);



%% Plot zz-plastic strain at tip element
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('ZZ plastic strain');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('ZZ-Plastic strain');

% Our result
t = axis_t;
ps = axis_pszz;
g = plot(t,ps);
set(g, 'Color', axis_color);


