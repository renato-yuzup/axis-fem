% Load our results
hypo_results;
E = 210e9;
nu = 0.3;
G = E / (2*(1+nu));
du = 625;

axis_color = [138 43 226] / 255;
axis2_color = [95 158 160] / 255;
axis3_color = [255 69 0] / 255;
axis4_color = [122 197 205] / 255;
theoretical1_color = [139 121 94] / 255;
theoretical2_color = [205 133 0] / 255;
theoretical3_color = [205 155 155] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

axis_t    = hypo_test(:,1);
axis_ux   = hypo_test(:,2);
axis_sxx  = hypo_test(:,3);
axis_szz  = hypo_test(:,4);
axis_sxz  = hypo_test(:,5);

% Create new figure;
f = figure;
set(f, 'Name', 'Test 5: Hypoelastic material II -- Comparative results', 'Color', [1 1 1]);


%% Plot displacement
h = subplot(1,2,1);
hold all;
set(h, 'FontSize', 8.0);
t = title('Top shear displacement');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

% Our result
t = axis_t;
u = axis_ux;
g = plot(t,u);
set(g, 'Color', axis_color);


%% Plot stress
h = subplot(1,2,2);
hold all;
set(h, 'FontSize', 8.0);
t = title('Element stress');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

% Theoretical s_xx
t = axis_t;
s = G * (1 - cos(du*t));
g = plot(t,s);
set(g, 'Color', theoretical1_color, 'LineStyle', '--', 'LineWidth', 3.0);

% Theoretical s_zz
t = axis_t;
s = -s;
g = plot(t,s);
set(g, 'Color', theoretical2_color, 'LineStyle', '--', 'LineWidth', 3.0);

% Theoretical s_xz
t = axis_t;
s = G * sin(du*t);
g = plot(t,s);
set(g, 'Color', theoretical3_color, 'LineStyle', '--', 'LineWidth', 3.0);

% Our result
t = axis_t;
s = axis_sxx;
g = plot(t,s);
set(g, 'Color', axis_color);

t = axis_t;
s = axis_szz;
g = plot(t,s);
set(g, 'Color', axis2_color);

t = axis_t;
s = axis_sxz;
g = plot(t,s);
set(g, 'Color', axis3_color);

legend({'\sigma_{xx} (theoretical)', '\sigma_{zz} (theoretical)', '\sigma_{xz} (theoretical)', '\sigma_{xx} (Axis)', '\sigma_{zz} (Axis)', '\sigma_{xz} (Axis)'}, 'Location', 'SouthEast');

