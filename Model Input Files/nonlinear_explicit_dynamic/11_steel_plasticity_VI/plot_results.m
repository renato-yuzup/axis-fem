% Load our results
plasticity_results;

% Load ABAQUS results
% abaqus_displacement_y;
% abaqus_stress_xx;
% abaqus_stress_yy;
% abaqus_stress_zz;
% abaqus_stress_yz;
% abaqus_stress_xz;
% abaqus_stress_xy;
% abaqus_plastic_strain_xx;
% abaqus_plastic_strain_yy;
% abaqus_plastic_strain_zz;
% abaqus_equiv_plastic_strain;

axis_color = [138 43 226] / 255;
abaqus_color = [124 205 124] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

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
set(f, 'Name', 'Test 9: Bilinear plasticity test IV -- Comparative results', 'Color', [1 1 1]);

%% Plot displacement at the free end
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Prescribed displacement at the free end (top)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement-Y (m)');

% Our result
t = axis_t;
u = axis_uy;
g = plot(t,u);
set(g, 'Color', axis_color);

% ABAQUS result
% t = abaqus_uy(:,1);
% u = abaqus_uy(:,2);
% g = plot(t,u);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');





% Create new figure;
f = figure;
set(f, 'Name', 'Test 9: Bilinear plasticity test IV -- Comparative results', 'Color', [1 1 1]);

%% Plot xx-stress
h = subplot(2,3,1);
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

% ABAQUS result
% t = abaqus_sigma_xx(:,1);
% sigma = abaqus_sigma_xx(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

%% Plot yy-stress
h = subplot(2,3,2);
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

% ABAQUS result
% t = abaqus_sigma_yy(:,1);
% sigma = abaqus_sigma_yy(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

%% Plot zz-stress
h = subplot(2,3,3);
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

% ABAQUS result
% t = abaqus_sigma_zz(:,1);
% sigma = abaqus_sigma_zz(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

%% Plot yz-stress
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('YZ stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Our result
t = axis_t;
sigma = axis_syz;
g = plot(t,sigma);
set(g, 'Color', axis_color);

% ABAQUS result
% t = abaqus_sigma_yz(:,1);
% sigma = abaqus_sigma_yz(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

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

% ABAQUS result
% t = abaqus_sigma_xz(:,1);
% sigma = abaqus_sigma_xz(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

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

% ABAQUS result
% t = abaqus_sigma_xy(:,1);
% sigma = abaqus_sigma_xy(:,2);
% g = plot(t,sigma);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');







% Create new figure;
f = figure;
set(f, 'Name', 'Test 9: Bilinear plasticity test IV -- Comparative results', 'Color', [1 1 1]);


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
set(g, 'Color', axis_color);

% ABAQUS result
% t = abaqus_equiv_pstrain(:,1);
% ep = abaqus_equiv_pstrain(:,2);
% g = plot(t,ep);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


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

% ABAQUS result
% t = abaqus_pstrain_xx(:,1);
% ps = abaqus_pstrain_xx(:,2);
% g = plot(t,ps);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


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

% ABAQUS result
% t = abaqus_pstrain_yy(:,1);
% ps = abaqus_pstrain_yy(:,2);
% g = plot(t,ps);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


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

% ABAQUS result
% t = abaqus_pstrain_zz(:,1);
% ps = abaqus_pstrain_zz(:,2);
% g = plot(t,ps);
% set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


