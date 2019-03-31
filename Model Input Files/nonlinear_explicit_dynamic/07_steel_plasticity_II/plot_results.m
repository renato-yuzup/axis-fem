% Load our results
plasticity_results;

% Load ABAQUS results
abaqus_stress_yy;
abaqus_stress_zz;
abaqus_plastic_strain_yy;
abaqus_equiv_pstrain;

axis_color = [138 43 226] / 255;
abaqus_color = [124 205 124] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

axis_t    = plastic_test(:,1);
axis_f    = plastic_test(:,2);
axis_uy   = plastic_test(:,3);
axis_sxx  = plastic_test(:,4);
axis_syy  = plastic_test(:,5);
axis_szz  = plastic_test(:,6);
axis_ep   = plastic_test(:,7);
axis_psxx = plastic_test(:,8);
axis_psyy = plastic_test(:,9);
axis_pszz = plastic_test(:,10);
axis_psyz = plastic_test(:,11);
axis_psxz = plastic_test(:,12);
axis_psxy = plastic_test(:,13);

% Create new figure;
f = figure;
set(f, 'Name', 'Test 7: Bilinear plasticity test I -- Comparative results', 'Color', [1 1 1]);


%% Plot yy-stress at tip element
h = subplot(2,3,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('YY stress at tip element');
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
t = abaqus_sigma_yy(:,1);
sigma = abaqus_sigma_yy(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot zz-stress at tip element
h = subplot(2,3,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('ZZ stress at tip element');
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
t = abaqus_sigma_zz(:,1);
sigma = abaqus_sigma_zz(:,2);
g = plot(t,sigma);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');



%% Plot equivalent plastic strain at tip element
h = subplot(2,3,4);
hold all;
set(h, 'FontSize', 8.0);
t = title('Equivalent plastic strain at tip element');
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
t = abaqus_epstrain(:,1);
ep = abaqus_epstrain(:,2);
g = plot(t,ep);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');


%% Plot yy-plastic strain at tip element
h = subplot(2,3,5);
hold all;
set(h, 'FontSize', 8.0);
t = title('YY plastic strain at tip element');
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
t = abaqus_pstrain_yy(:,1);
ps = abaqus_pstrain_yy(:,2);
g = plot(t,ps);
set(g, 'Color', abaqus_color);

legend({'Axis', 'ABAQUS'}, 'Location', 'SouthEast');

