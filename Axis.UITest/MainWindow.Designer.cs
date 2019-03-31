namespace Axis.UITest
{
    partial class MainWindow
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.label1 = new System.Windows.Forms.Label();
            this.txtInputFile = new System.Windows.Forms.TextBox();
            this.cmdBrowseInputFile = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.txtIncludeFolder = new System.Windows.Forms.TextBox();
            this.cmdBrowseIncludeFolder = new System.Windows.Forms.Button();
            this.cmdBrowseOutputFolder = new System.Windows.Forms.Button();
            this.txtOutputFolder = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.dlgBrowseInputFile = new System.Windows.Forms.OpenFileDialog();
            this.dlgBrowseIncludeFolder = new System.Windows.Forms.FolderBrowserDialog();
            this.dlgBrowseOutputFolder = new System.Windows.Forms.FolderBrowserDialog();
            this.ttError = new System.Windows.Forms.ToolTip(this.components);
            this.lstFlags = new System.Windows.Forms.ListBox();
            this.label4 = new System.Windows.Forms.Label();
            this.txtFlag = new System.Windows.Forms.TextBox();
            this.cmdAddFlag = new System.Windows.Forms.Button();
            this.cmdRemoveFlag = new System.Windows.Forms.Button();
            this.cboHardwareSelect = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.cmdSubmit = new System.Windows.Forms.Button();
            this.cmdExit = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(18, 66);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(56, 14);
            this.label1.TabIndex = 0;
            this.label1.Text = "Input file";
            // 
            // txtInputFile
            // 
            this.txtInputFile.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtInputFile.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
            this.txtInputFile.Location = new System.Drawing.Point(158, 63);
            this.txtInputFile.Name = "txtInputFile";
            this.txtInputFile.Size = new System.Drawing.Size(298, 22);
            this.txtInputFile.TabIndex = 1;
            this.toolTip1.SetToolTip(this.txtInputFile, "Select the input file from which model and analysis \r\ninformation will be read.");
            this.txtInputFile.TextChanged += new System.EventHandler(this.InputFile_TextChanged);
            this.txtInputFile.Validating += new System.ComponentModel.CancelEventHandler(this.InputFile_OnValidate);
            // 
            // cmdBrowseInputFile
            // 
            this.cmdBrowseInputFile.Location = new System.Drawing.Point(459, 61);
            this.cmdBrowseInputFile.Name = "cmdBrowseInputFile";
            this.cmdBrowseInputFile.Size = new System.Drawing.Size(75, 25);
            this.cmdBrowseInputFile.TabIndex = 2;
            this.cmdBrowseInputFile.Text = "Browse";
            this.cmdBrowseInputFile.UseVisualStyleBackColor = true;
            this.cmdBrowseInputFile.Click += new System.EventHandler(this.cmdBrowseInputFile_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(18, 94);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(134, 14);
            this.label2.TabIndex = 3;
            this.label2.Text = "Include file base folder";
            // 
            // txtIncludeFolder
            // 
            this.txtIncludeFolder.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtIncludeFolder.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystemDirectories;
            this.txtIncludeFolder.Location = new System.Drawing.Point(158, 91);
            this.txtIncludeFolder.Name = "txtIncludeFolder";
            this.txtIncludeFolder.Size = new System.Drawing.Size(298, 22);
            this.txtIncludeFolder.TabIndex = 4;
            this.toolTip1.SetToolTip(this.txtIncludeFolder, "If your analysis uses more than one input file, select a folder \r\nfrom where sear" +
        "ch for included files will begin.\r\n\r\nRelative paths used to reference include fi" +
        "les will use this \r\nbase folder.");
            this.txtIncludeFolder.TextChanged += new System.EventHandler(this.txtIncludeFolder_TextChanged);
            // 
            // cmdBrowseIncludeFolder
            // 
            this.cmdBrowseIncludeFolder.Location = new System.Drawing.Point(459, 90);
            this.cmdBrowseIncludeFolder.Name = "cmdBrowseIncludeFolder";
            this.cmdBrowseIncludeFolder.Size = new System.Drawing.Size(75, 23);
            this.cmdBrowseIncludeFolder.TabIndex = 5;
            this.cmdBrowseIncludeFolder.Text = "Browse";
            this.cmdBrowseIncludeFolder.UseVisualStyleBackColor = true;
            // 
            // cmdBrowseOutputFolder
            // 
            this.cmdBrowseOutputFolder.Location = new System.Drawing.Point(459, 118);
            this.cmdBrowseOutputFolder.Name = "cmdBrowseOutputFolder";
            this.cmdBrowseOutputFolder.Size = new System.Drawing.Size(75, 23);
            this.cmdBrowseOutputFolder.TabIndex = 8;
            this.cmdBrowseOutputFolder.Text = "Browse";
            this.cmdBrowseOutputFolder.UseVisualStyleBackColor = true;
            // 
            // txtOutputFolder
            // 
            this.txtOutputFolder.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtOutputFolder.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystemDirectories;
            this.txtOutputFolder.Location = new System.Drawing.Point(158, 119);
            this.txtOutputFolder.Name = "txtOutputFolder";
            this.txtOutputFolder.Size = new System.Drawing.Size(298, 22);
            this.txtOutputFolder.TabIndex = 7;
            this.toolTip1.SetToolTip(this.txtOutputFolder, "Select the folder where output files will be written.");
            this.txtOutputFolder.TextChanged += new System.EventHandler(this.txtOutputFolder_TextChanged);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(18, 122);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(80, 14);
            this.label3.TabIndex = 6;
            this.label3.Text = "Output folder";
            // 
            // toolTip1
            // 
            this.toolTip1.IsBalloon = true;
            this.toolTip1.ToolTipIcon = System.Windows.Forms.ToolTipIcon.Info;
            this.toolTip1.ToolTipTitle = "Help";
            // 
            // dlgBrowseInputFile
            // 
            this.dlgBrowseInputFile.DefaultExt = "axis";
            this.dlgBrowseInputFile.Filter = "Axis Input Files|*.axis|All files|*.*";
            // 
            // ttError
            // 
            this.ttError.IsBalloon = true;
            this.ttError.ToolTipIcon = System.Windows.Forms.ToolTipIcon.Error;
            this.ttError.ToolTipTitle = "Invalid information";
            // 
            // lstFlags
            // 
            this.lstFlags.FormattingEnabled = true;
            this.lstFlags.ItemHeight = 14;
            this.lstFlags.Location = new System.Drawing.Point(255, 223);
            this.lstFlags.Name = "lstFlags";
            this.lstFlags.Size = new System.Drawing.Size(197, 130);
            this.lstFlags.Sorted = true;
            this.lstFlags.TabIndex = 13;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(252, 178);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(115, 14);
            this.label4.TabIndex = 11;
            this.label4.Text = "Active analysis flags";
            // 
            // txtFlag
            // 
            this.txtFlag.Location = new System.Drawing.Point(255, 195);
            this.txtFlag.Name = "txtFlag";
            this.txtFlag.Size = new System.Drawing.Size(197, 22);
            this.txtFlag.TabIndex = 12;
            this.txtFlag.Enter += new System.EventHandler(this.txtFlag_OnFocus);
            this.txtFlag.Leave += new System.EventHandler(this.txtFlag_OnLeave);
            // 
            // cmdAddFlag
            // 
            this.cmdAddFlag.Location = new System.Drawing.Point(458, 194);
            this.cmdAddFlag.Name = "cmdAddFlag";
            this.cmdAddFlag.Size = new System.Drawing.Size(75, 23);
            this.cmdAddFlag.TabIndex = 14;
            this.cmdAddFlag.Text = "Add";
            this.cmdAddFlag.UseVisualStyleBackColor = true;
            this.cmdAddFlag.Click += new System.EventHandler(this.cmdAddFlag_Click);
            // 
            // cmdRemoveFlag
            // 
            this.cmdRemoveFlag.Location = new System.Drawing.Point(458, 223);
            this.cmdRemoveFlag.Name = "cmdRemoveFlag";
            this.cmdRemoveFlag.Size = new System.Drawing.Size(75, 23);
            this.cmdRemoveFlag.TabIndex = 15;
            this.cmdRemoveFlag.Text = "Remove";
            this.cmdRemoveFlag.UseVisualStyleBackColor = true;
            this.cmdRemoveFlag.Click += new System.EventHandler(this.cmdRemoveFlag_Click);
            // 
            // cboHardwareSelect
            // 
            this.cboHardwareSelect.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboHardwareSelect.FormattingEnabled = true;
            this.cboHardwareSelect.Items.AddRange(new object[] {
            "Use environment settings",
            "Force processing on CPU",
            "Force processing on GPU"});
            this.cboHardwareSelect.Location = new System.Drawing.Point(21, 195);
            this.cboHardwareSelect.Name = "cboHardwareSelect";
            this.cboHardwareSelect.Size = new System.Drawing.Size(184, 22);
            this.cboHardwareSelect.TabIndex = 10;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(18, 178);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(120, 14);
            this.label5.TabIndex = 9;
            this.label5.Text = "Processing hardware";
            // 
            // cmdSubmit
            // 
            this.cmdSubmit.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.cmdSubmit.Location = new System.Drawing.Point(357, 389);
            this.cmdSubmit.Name = "cmdSubmit";
            this.cmdSubmit.Size = new System.Drawing.Size(85, 30);
            this.cmdSubmit.TabIndex = 16;
            this.cmdSubmit.Text = "Submit";
            this.cmdSubmit.UseVisualStyleBackColor = true;
            // 
            // cmdExit
            // 
            this.cmdExit.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cmdExit.Location = new System.Drawing.Point(448, 389);
            this.cmdExit.Name = "cmdExit";
            this.cmdExit.Size = new System.Drawing.Size(85, 30);
            this.cmdExit.TabIndex = 17;
            this.cmdExit.Text = "Exit";
            this.cmdExit.UseVisualStyleBackColor = true;
            this.cmdExit.Click += new System.EventHandler(this.cmdExit_Click);
            // 
            // MainWindow
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 14F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.cmdExit;
            this.ClientSize = new System.Drawing.Size(551, 431);
            this.Controls.Add(this.cmdExit);
            this.Controls.Add(this.cmdSubmit);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.cboHardwareSelect);
            this.Controls.Add(this.cmdRemoveFlag);
            this.Controls.Add(this.cmdAddFlag);
            this.Controls.Add(this.txtFlag);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.lstFlags);
            this.Controls.Add(this.cmdBrowseOutputFolder);
            this.Controls.Add(this.txtOutputFolder);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.cmdBrowseIncludeFolder);
            this.Controls.Add(this.txtIncludeFolder);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.cmdBrowseInputFile);
            this.Controls.Add(this.txtInputFile);
            this.Controls.Add(this.label1);
            this.Font = new System.Drawing.Font("Calibri", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "MainWindow";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Axis Solver: Concept UI";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtInputFile;
        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.Button cmdBrowseInputFile;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtIncludeFolder;
        private System.Windows.Forms.Button cmdBrowseIncludeFolder;
        private System.Windows.Forms.Button cmdBrowseOutputFolder;
        private System.Windows.Forms.TextBox txtOutputFolder;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.OpenFileDialog dlgBrowseInputFile;
        private System.Windows.Forms.FolderBrowserDialog dlgBrowseIncludeFolder;
        private System.Windows.Forms.FolderBrowserDialog dlgBrowseOutputFolder;
        private System.Windows.Forms.ToolTip ttError;
        private System.Windows.Forms.ListBox lstFlags;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtFlag;
        private System.Windows.Forms.Button cmdAddFlag;
        private System.Windows.Forms.Button cmdRemoveFlag;
        private System.Windows.Forms.ComboBox cboHardwareSelect;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Button cmdSubmit;
        private System.Windows.Forms.Button cmdExit;
    }
}

