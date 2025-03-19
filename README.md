<h1 align="center">🔒 Privacy_Preserving_Data_Analysis 🔒</h1>
<h3 align="center">Hybrid Model for CIFAR-10 with Performance and Efficiency Optimization</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Hybrid-blue?style=for-the-badge" alt="Model Type" />
  <img src="https://img.shields.io/badge/Dataset-CIFAR--10-orange?style=for-the-badge" alt="Dataset" />
  <img src="https://img.shields.io/badge/Accuracy-68%25-green?style=for-the-badge" alt="Accuracy" />
</p>

---

<h2>📌 Overview</h2>

<p>
  This project develops a HybridNet classifier for the CIFAR-10 dataset, designed to achieve <b>high accuracy</b> under <b>strict differential privacy (DP)</b> constraints.  
  The goal: <b>Maximize performance</b> while <b>minimizing computational resources</b> and <b>maintaining privacy</b> (ε ≤ 3, δ = 10⁻⁵).  
</p>

---

<h2>🚀 Model Architecture</h2>

<p>We strategically integrated the strengths of top-performing architectures to form HybridNet:</p>

<ul>
  <li><b>Convolutional Neural Network (ConvNet)</b>: Captures spatial patterns effectively.</li>
  <li><b>ResNet (Residual Network)</b>: Helps combat vanishing gradients with skip connections.</li>
  <li><b>TAN Code (Truncated Affine Network)</b>: Speeds up execution while maintaining accuracy.</li>
  <li><b>DenseNet (Densely Connected Network)</b>: Enhances feature propagation and encourages feature reuse.</li>
  <li><b>GoogLeNet (Inception Network)</b>: Employs multiple filter sizes simultaneously for diversified learning.</li>
</ul>

<p>⚡️ Additionally, <b>skip connections</b> were strategically incorporated to improve gradient flow and optimize convergence speed.</p>

---

<h2>📊 Results</h2>

<p><b>✅ Final Results (ε ≤ 3):</b></p>

<ul>
  <li>🔥 <b>ResNet32:</b> Achieved the highest accuracy of <b>65.41%</b></li>
  <li>🚀 <b>DenseNet:</b> Followed closely with an accuracy of <b>62.71%</b></li>
  <li>🔧 <b>EfficientNet (Experimental):</b> Struggled with memory issues, achieving <b>39.5%</b> accuracy</li>
</ul>

<p>📈 Previous models (ResNet, DenseNet) suffered from poor accuracy (~26-60%) under stricter privacy budgets (ε ≤ 3). HybridNet significantly improves performance without exceeding privacy constraints.</p>


<p><b>Key Results:</b></p>
<ul>
  <li>✅ Balanced accuracy and speed with less resource usage.</li>
  <li>✅ Effective use of skip connections for faster convergence.</li>
  <li>✅ Demonstrated potential for further performance improvement.</li>
</ul>

---

<h2>⚙️ How to Run the Code</h2>

<p>To train the model, use the following command:</p>

<pre><code>python3 train_tan_{model_name}.py --batch_size 256 --ref_nb_steps 875 --ref_B 4096 --ref_noise 3 --transform 16 --data_root "path to load or store CIFAR10"</code></pre>

<p><b>Instructions:</b></p>
<ul>
  <li>Replace <code>{model_name}</code> with your desired model (e.g., <code>convnet</code>, <code>resnet</code>, <code>densenet</code>).</li>
  <li>Ensure the <code>--data_root</code> points to the correct path for CIFAR-10 data.</li>
</ul>

---

<h2>🔧 Set Epsilon Values</h2>

<p>To adjust the model’s privacy parameter (ε) based on DeepMind's recommendations, modify <code>--ref_nb_steps</code> as shown:</p>

<ul>
  <li>🔹 <b>ε = 1</b>: <code>--ref_nb_steps 875</code></li>
  <li>🔹 <b>ε = 2</b>: <code>--ref_nb_steps 1125</code></li>
  <li>🔹 <b>ε = 3</b>: <code>--ref_nb_steps 1593</code></li>
</ul>

<p>✨ <b>Tip:</b> Higher epsilon allows less privacy but improves accuracy.</p>

---

<h2>🔍 Future Improvements</h2>

<p>We identified key areas to push the model's performance even further:</p>

<ul>
  <li>🔧 Implement advanced data augmentation techniques.</li>
  <li>🔧 Fine-tune hyperparameters and optimize learning rates.</li>
  <li>🔧 Experiment with newer architectures (e.g., Vision Transformers).</li>
  <li>🔧 Explore pruning and quantization for even more resource savings.</li>
</ul>

---

<h2>📂 Dataset</h2>

<p>The model is trained on the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10 dataset</a>, which consists of:</p>

<ul>
  <li>50,000 training images (10 classes)</li>
  <li>10,000 test images (10 classes)</li>
  <li>Image size: 32x32 pixels (color)</li>
</ul>

---

<h2>🔧 Technologies Used</h2>

<p>
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-red?style=flat-square" alt="PyTorch" />
  <img src="https://img.shields.io/badge/NumPy-green?style=flat-square" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-blue?style=flat-square" alt="Pandas" />
  <img src="https://img.shields.io/badge/OpenCV-darkgreen?style=flat-square" alt="OpenCV" />
  <img src="https://img.shields.io/badge/SciPy-orange?style=flat-square" alt="SciPy" />
</p>

---

<h2>📫 Contact</h2>

<p>
  If you want to dive deeper into the project, feel free to reach out to me:
  <br>
  📧 Email: <b>gantisharan6639@gmail.com</b>  
  🔗 LinkedIn: <a href="https://linkedin.com/in/sharan-ganti" target="_blank">Sharan Ganti</a>
</p>

---

<p align="center">💪🏼 Built with passion, performance, and privacy in mind! 💪🏼</p>
