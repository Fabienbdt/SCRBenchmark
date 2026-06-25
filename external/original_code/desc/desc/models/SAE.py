import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
import  math
import numpy as np
import random
import tensorflow as tf
#random.seed(201809)
#np.random.seed(201809)
#tf.set_random_seed(201809) if tf.__version__<="2.0" else tf.random.set_seed(201809)

class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model.
        random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed
        actincenter: the activation function in last layer for encoder and last layer for encoder (avoiding the representation values and reconstruct outputs are all non-negative)
        init: `str`,optional. Default: `glorot_uniform`. Initialization method used to initialize weights.
        use_earlyStop: optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.
        save_dir:'str',optional. Default,'result_tmp',some result will be saved in this directory.
    """
    def __init__(self, dims, act='relu', 
            drop_rate=0.2, 
            batch_size=32,
            random_seed=201809,
            actincenter="tanh",
            init="glorot_uniform",
            use_earlyStop=True,
            save_dir='result_tmp',
            weighted_training=False,
            warmup_epochs=30,
            dynamic_weight_update_interval=10,
            dynamic_weight_momentum=0.7,
            pseudo_label_method="leiden",
            weight_exponent=0.2,
            weight_n_clusters=0,
            density_knn_k=15,
            density_weight_exponent=1.0,
            density_weight_clip=5.0,
            weight_fusion_mode="additive",
            cluster_density_alpha=0.6,
            cluster_weight_power=1.0,
            density_weight_power=1.0,
            min_cell_weight=0.25,
            max_cell_weight=10.0,
            weight_component_mode="full",
            rare_triplet_weight=0.0,
            rare_triplet_margin=0.4,
            rare_triplet_min_weight=1.2,
            rare_triplet_start_epoch=60,
            max_triplet_anchors_per_batch=64,
            triplet_pseudo_label_method="kmeans"): #act relu
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.actincenter=actincenter #linear
        self.drop_rate = drop_rate
        self.init=init
        self.batch_size = batch_size
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        #tf.set_random_seed(random_seed)
        tf.set_random_seed(random_seed) if tf.__version__<"2.0" else tf.random.set_seed(random_seed)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #
        self.random_seed=random_seed
        self.use_earlyStop=use_earlyStop
        self.weighted_training = bool(weighted_training)
        self.warmup_epochs = int(warmup_epochs)
        self.dynamic_weight_update_interval = int(dynamic_weight_update_interval)
        self.dynamic_weight_momentum = float(dynamic_weight_momentum)
        self.pseudo_label_method = str(pseudo_label_method)
        self.weight_exponent = float(weight_exponent)
        self.weight_n_clusters = int(weight_n_clusters)
        self.density_knn_k = int(density_knn_k)
        self.density_weight_exponent = float(density_weight_exponent)
        self.density_weight_clip = float(density_weight_clip)
        self.weight_fusion_mode = str(weight_fusion_mode)
        self.cluster_density_alpha = float(cluster_density_alpha)
        self.cluster_weight_power = float(cluster_weight_power)
        self.density_weight_power = float(density_weight_power)
        self.min_cell_weight = float(min_cell_weight)
        self.max_cell_weight = float(max_cell_weight)
        self.weight_component_mode = str(weight_component_mode)
        self.rare_triplet_weight = float(rare_triplet_weight)
        self.rare_triplet_margin = float(rare_triplet_margin)
        self.rare_triplet_min_weight = float(rare_triplet_min_weight)
        self.rare_triplet_start_epoch = int(rare_triplet_start_epoch)
        self.max_triplet_anchors_per_batch = int(max_triplet_anchors_per_batch)
        self.triplet_pseudo_label_method = str(triplet_pseudo_label_method)
        self.pretrain_loss_history = []
        self.pretrain_recon_history = []
        self.pretrain_triplet_history = []
        self.pretrain_triplet_valid_anchor_history = []
        self.weight_history = []
        self.stacks = [self.make_stack(i,random_seed=self.random_seed+2*i) for i in range(self.n_stacks)]
        self.autoencoders ,self.encoder= self.make_autoencoders()
        #plot_model(self.autoencoders, show_shapes=True, to_file=os.path.join(save_dir,'autoencoders.png'))

    def _summarize_cell_weights(self, weights, epoch, phase, refreshed):
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        row = {
            "epoch": int(epoch),
            "phase": str(phase),
            "refreshed": bool(refreshed),
        }
        if w.size == 0:
            row.update({
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "max": float("nan"),
            })
            return row
        row.update({
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "p05": float(np.percentile(w, 5)),
            "p50": float(np.percentile(w, 50)),
            "p95": float(np.percentile(w, 95)),
            "max": float(np.max(w)),
        })
        return row
    def choose_init(self,init="glorot_uniform",seed=1):
        if init not in {'glorot_uniform','glorot_normal','he_normal','lecun_normal','he_uniform','lecun_uniform','RandomNormal','RandomUniform',"TruncatedNormal"}:
            raise ValueError('Invalid `init` argument: '
                             'expected on of {"glorot_uniform", "glorot_normal", "he_normal","he_uniform","lecun_normal","lecun_uniform","RandomNormal","RandomUniform","TruncatedNormal"} '
                             'but got', mode)
        """
        #tensorflow <2.0
        if init=="glorot_uniform":
            res=keras.initializers.glorot_uniform(seed=seed)
        elif init=="glorot_normal":
            res=keras.initializers.glorot_normal(seed=seed)
        elif init=="he_normal":
            res=keras.initializers.he_normal(seed=seed)
        elif init=='he_uniform':
            res=keras.initializers.he_uniform(seed=seed)
        elif init=="lecun_normal":
            res=keras.initializer.lecun_normal(seed=seed)
        elif init=="lecun_uniform":
            res=keras.initializers.lecun_uniform(seed=seed)
        elif init=="RandomNormal":
            res=keras.initializers.RandomNormal(mean=0.0,stddev=0.04,seed=seed)
        elif init=="RandomUniform":
            res=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05,seed=seed)
        else:
            res=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed)
        """
        return init
        
        

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        # input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+i),activation=self.activation, name='encoder_%d' % i)(h)

        # hidden layer,default activation is linear
        h = Dense(self.dims[-1],kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks), name='encoder_%d' % (self.n_stacks - 1),activation=self.actincenter)(h)  # features are extracted from here

        y=h
        # internal layers in decoder       
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks+i),activation=self.activation, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+2*self.n_stacks),name='decoder_0',activation=self.actincenter)(y)

        return Model(inputs=x, outputs=y,name="AE"),Model(inputs=x,outputs=h,name="encoder")

    def make_stack(self, ith,random_seed=0):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = self.actincenter# tanh, or linear
        if ith == self.n_stacks-1:
            hidden_act = self.actincenter #tanh, or linear
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,),seed=random_seed))
        model.add(Dense(units=hidden_dim, activation=hidden_act, kernel_initializer=self.choose_init(init=self.init,seed=random_seed),name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate,seed=random_seed+1))
        model.add(Dense(units=in_out_dim, activation=output_act,kernel_initializer=self.choose_init(init=self.init,seed=random_seed+1), name='decoder_%d' % ith))
        return model

    def _sanitize_embeddings(self, values, context):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        non_finite = ~np.isfinite(arr)
        if np.any(non_finite):
            print(
                "SAE weighted pretraining: found",
                int(np.sum(non_finite)),
                "non-finite values in",
                context,
                "- applying nan_to_num",
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        return np.clip(arr, -1e4, 1e4).astype(np.float32, copy=False)

    def _resolve_weight_k(self, n_cells):
        if self.weight_n_clusters > 1:
            return int(max(2, min(self.weight_n_clusters, n_cells)))
        return int(max(2, min(8, n_cells)))

    def _weight_component_mode(self):
        mode = str(self.weight_component_mode).strip().lower()
        if mode not in {"full", "density_only"}:
            print("SAE weighted pretraining: unknown weight_component_mode=", mode, "using full")
            mode = "full"
        return mode

    def _leiden_pseudo_labels(self, embeddings, n_clusters):
        import anndata as ad
        import scanpy as sc

        emb = self._sanitize_embeddings(embeddings, "leiden pseudo-label input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.zeros(n_cells, dtype=np.int64)

        target_k = int(max(2, min(int(n_clusters), n_cells)))
        adata = ad.AnnData(X=emb)
        sc.pp.neighbors(adata, n_neighbors=min(15, max(2, n_cells - 1)), use_rep="X")

        best_res = 1.0
        best_diff = n_cells
        for res in np.arange(0.05, 3.0, 0.05):
            sc.tl.leiden(adata, resolution=float(res), random_state=int(self.random_seed))
            n_found = len(np.unique(adata.obs["leiden"].astype(int).values))
            diff = abs(n_found - target_k)
            if diff < best_diff:
                best_diff = diff
                best_res = float(res)
            if n_found == target_k:
                break

        sc.tl.leiden(adata, resolution=best_res, random_state=int(self.random_seed))
        return adata.obs["leiden"].astype(int).values.astype(np.int64)

    def _compute_cluster_frequency_weights(self, embeddings):
        from sklearn.cluster import KMeans

        emb = self._sanitize_embeddings(embeddings, "cluster-frequency input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.ones(n_cells, dtype=np.float32), np.zeros(n_cells, dtype=np.int64)

        k = self._resolve_weight_k(n_cells)
        method = str(self.pseudo_label_method).strip().lower()
        exponent = float(self.weight_exponent)

        if method == "leiden":
            pseudo_labels = self._leiden_pseudo_labels(emb, k)
        else:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=int(self.random_seed))
            pseudo_labels = kmeans.fit_predict(emb).astype(np.int64)

        unique_labels, counts = np.unique(pseudo_labels, return_counts=True)
        freqs = counts / np.sum(counts)
        label_to_weight = {
            int(label): float((1.0 / max(freq, 1e-8)) ** exponent)
            for label, freq in zip(unique_labels, freqs)
        }
        weights = np.array([label_to_weight[int(label)] for label in pseudo_labels], dtype=np.float32)
        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights = weights / mean_weight
        return weights.astype(np.float32), pseudo_labels.astype(np.int64)

    def _pseudo_labels_for_method(self, embeddings, method):
        from sklearn.cluster import KMeans

        emb = self._sanitize_embeddings(embeddings, str(method) + " pseudo-label input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.zeros(n_cells, dtype=np.int64)

        k = self._resolve_weight_k(n_cells)
        method = str(method).strip().lower()
        if method == "leiden":
            return self._leiden_pseudo_labels(emb, k)
        if method != "kmeans":
            print("SAE weighted pretraining: unknown pseudo-label method", method, "using kmeans")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=int(self.random_seed))
        return kmeans.fit_predict(emb).astype(np.int64)

    def _compute_density_weights(self, embeddings):
        from sklearn.neighbors import NearestNeighbors

        emb = self._sanitize_embeddings(embeddings, "density input")
        n_cells = emb.shape[0]
        if n_cells <= 2:
            return np.ones(n_cells, dtype=np.float32)

        k = int(max(2, min(int(self.density_knn_k), n_cells - 1)))
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(emb)
        distances, _ = nn.kneighbors(emb)

        kth_dist = distances[:, -1].astype(np.float32)
        scale = float(np.median(kth_dist)) + 1e-8
        normalized = kth_dist / scale

        weights = np.power(np.maximum(normalized, 1e-8), float(self.density_weight_exponent))
        weights = np.clip(weights, 0.05, float(self.density_weight_clip)).astype(np.float32)
        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights = weights / mean_weight
        return weights.astype(np.float32)

    def _compute_combined_cell_weights(self, embeddings):
        density_weights = self._compute_density_weights(embeddings)
        if self._weight_component_mode() == "density_only":
            pseudo_labels = np.zeros(np.asarray(embeddings).shape[0], dtype=np.int64)
            min_cell_weight = min(float(self.min_cell_weight), float(self.max_cell_weight))
            max_cell_weight = float(self.max_cell_weight)
            weights = np.clip(density_weights, min_cell_weight, max_cell_weight).astype(np.float32)
            mean_weight = float(np.mean(weights))
            if mean_weight > 0:
                weights = weights / mean_weight
            return weights.astype(np.float32), pseudo_labels

        cluster_weights, pseudo_labels = self._compute_cluster_frequency_weights(embeddings)

        cw = np.power(np.maximum(cluster_weights, 1e-8), float(self.cluster_weight_power))
        dw = np.power(np.maximum(density_weights, 1e-8), float(self.density_weight_power))

        if str(self.weight_fusion_mode).strip().lower() == "multiplicative":
            combined = (cw * dw).astype(np.float32)
        else:
            alpha = float(self.cluster_density_alpha)
            combined = (alpha * cw + (1.0 - alpha) * dw).astype(np.float32)

        mean_weight = float(np.mean(combined))
        if mean_weight > 0:
            combined = combined / mean_weight

        min_cell_weight = min(float(self.min_cell_weight), float(self.max_cell_weight))
        max_cell_weight = float(self.max_cell_weight)
        combined = np.clip(combined, min_cell_weight, max_cell_weight).astype(np.float32)
        return combined, pseudo_labels

    def _compute_triplet_pseudo_labels(self, embeddings, fallback_labels=None):
        method = str(self.triplet_pseudo_label_method).strip().lower()
        if (
            fallback_labels is not None
            and self._weight_component_mode() == "full"
            and method == str(self.pseudo_label_method).strip().lower()
        ):
            return np.asarray(fallback_labels, dtype=np.int64)
        return self._pseudo_labels_for_method(embeddings, method)

    def _tf_pairwise_distances(self, z):
        dot = tf.matmul(z, z, transpose_b=True)
        square_norm = tf.linalg.diag_part(dot)
        distances = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot
            + tf.expand_dims(square_norm, 0)
        )
        distances = tf.maximum(distances, 0.0)
        return tf.sqrt(distances + 1e-12)

    def _tf_rare_triplet_loss(self, z, batch_weights, batch_labels):
        z = tf.convert_to_tensor(z)
        batch_weights = tf.reshape(tf.cast(batch_weights, z.dtype), [-1])
        batch_labels = tf.reshape(tf.cast(batch_labels, tf.int64), [-1])
        n_batch = tf.shape(z)[0]
        if z.shape[0] is not None and z.shape[0] < 3:
            return tf.zeros([], dtype=z.dtype), tf.constant(0.0, dtype=z.dtype)

        dists = self._tf_pairwise_distances(z)
        labels_equal = tf.equal(tf.expand_dims(batch_labels, 1), tf.expand_dims(batch_labels, 0))
        eye = tf.eye(n_batch, dtype=tf.bool)
        same = tf.logical_and(labels_equal, tf.logical_not(eye))
        diff = tf.logical_and(tf.logical_not(labels_equal), tf.logical_not(eye))

        large = tf.constant(1e6, dtype=z.dtype)
        d_pos = tf.reduce_max(tf.where(same, dists, tf.zeros_like(dists)), axis=1)
        has_pos = tf.reduce_any(same, axis=1)
        has_neg = tf.reduce_any(diff, axis=1)

        neg_dists = tf.where(diff, dists, tf.ones_like(dists) * large)
        semi_hard = tf.logical_and(diff, dists > tf.expand_dims(d_pos, 1))
        semi_neg_dists = tf.where(semi_hard, dists, tf.ones_like(dists) * large)
        min_semi = tf.reduce_min(semi_neg_dists, axis=1)
        min_neg = tf.reduce_min(neg_dists, axis=1)
        d_neg = tf.where(min_semi < large, min_semi, min_neg)

        candidate = tf.logical_and(
            batch_weights >= float(self.rare_triplet_min_weight),
            tf.logical_and(has_pos, has_neg),
        )
        candidate_idx = tf.reshape(tf.where(candidate), [-1])
        max_anchors = int(self.max_triplet_anchors_per_batch)
        if max_anchors > 0:
            candidate_idx = candidate_idx[:max_anchors]

        selected_pos = tf.gather(d_pos, candidate_idx)
        selected_neg = tf.gather(d_neg, candidate_idx)
        losses = tf.nn.relu(selected_pos - selected_neg + float(self.rare_triplet_margin))
        valid = tf.cast(tf.size(losses), z.dtype)
        return tf.cond(
            tf.size(losses) > 0,
            lambda: (tf.reduce_mean(losses), valid),
            lambda: (tf.zeros([], dtype=z.dtype), tf.constant(0.0, dtype=z.dtype)),
        )

    def _refresh_cell_weights(self, x, previous_weights, epoch):
        features = self.encoder.predict(x, verbose=0)
        fresh_weights, pseudo_labels = self._compute_combined_cell_weights(features)

        if previous_weights is not None:
            momentum = float(np.clip(self.dynamic_weight_momentum, 0.0, 0.95))
            mixed = momentum * previous_weights + (1.0 - momentum) * fresh_weights
            mean_weight = float(np.mean(mixed))
            if mean_weight > 0:
                mixed = mixed / mean_weight
            fresh_weights = mixed.astype(np.float32)

        min_cell_weight = min(float(self.min_cell_weight), float(self.max_cell_weight))
        max_cell_weight = float(self.max_cell_weight)
        fresh_weights = np.clip(fresh_weights, min_cell_weight, max_cell_weight).astype(np.float32)

        stats = self._summarize_cell_weights(
            fresh_weights,
            epoch=epoch,
            phase="weighted",
            refreshed=True,
        )
        print(
            "SAE weighted pretraining: refreshed weights at epoch",
            int(epoch),
            "| min=%.3f p50=%.3f p95=%.3f max=%.3f"
            % (stats["min"], stats["p50"], stats["p95"], stats["max"]),
        )
        return fresh_weights, pseudo_labels

    def _fit_autoencoder_weighted(self, x, epochs=300):
        x = np.asarray(x, dtype=np.float32)
        n_samples = x.shape[0]
        if n_samples == 0:
            return

        warmup_epochs = int(max(0, min(int(self.warmup_epochs), int(epochs))))
        update_interval = int(max(0, int(self.dynamic_weight_update_interval)))
        batch_size = int(max(1, self.batch_size))
        current_weights = np.ones(n_samples, dtype=np.float32)
        current_pseudo_labels = np.zeros(n_samples, dtype=np.int64)
        current_triplet_labels = np.zeros(n_samples, dtype=np.int64)
        best_loss = np.inf
        patience = 0
        stop_training = False

        print('Fine-tuning autoencoder end-to-end (scRAW-weighted)')
        total_chunks = int(math.ceil(float(epochs) / 50.0))
        for j in range(total_chunks):
            lr = pow(10, -j)
            start_epoch = j * 50
            chunk_epochs = min(50, int(epochs) - start_epoch)
            if chunk_epochs <= 0:
                break

            print('learning rate =', lr)
            optimizer = SGD(lr, momentum=0.9)
            self.autoencoders.compile(optimizer=optimizer, loss='mse')
            triplet_model = Model(inputs=self.autoencoders.input, outputs=[self.autoencoders.output, self.encoder.output])
            for _ in range(chunk_epochs):
                current_epoch = len(self.pretrain_loss_history)
                weights_refreshed = False
                if current_epoch == warmup_epochs and current_epoch < int(epochs):
                    current_weights, current_pseudo_labels = self._refresh_cell_weights(
                        x,
                        previous_weights=None,
                        epoch=current_epoch,
                    )
                    weights_refreshed = True
                    if self.rare_triplet_weight > 0:
                        features = self.encoder.predict(x, verbose=0)
                        current_triplet_labels = self._compute_triplet_pseudo_labels(
                            features,
                            fallback_labels=current_pseudo_labels,
                        )
                elif (
                    current_epoch > warmup_epochs
                    and update_interval > 0
                    and ((current_epoch - warmup_epochs) % update_interval == 0)
                ):
                    current_weights, current_pseudo_labels = self._refresh_cell_weights(
                        x,
                        previous_weights=current_weights.copy(),
                        epoch=current_epoch,
                    )
                    weights_refreshed = True
                    if self.rare_triplet_weight > 0:
                        features = self.encoder.predict(x, verbose=0)
                        current_triplet_labels = self._compute_triplet_pseudo_labels(
                            features,
                            fallback_labels=current_pseudo_labels,
                        )

                order = np.random.permutation(n_samples)
                epoch_loss = 0.0
                epoch_recon = 0.0
                epoch_triplet = 0.0
                epoch_valid = 0.0
                for batch_start in range(0, n_samples, batch_size):
                    idx = order[batch_start: batch_start + batch_size]
                    batch_x = x[idx]
                    triplet_enabled = (
                        self.rare_triplet_weight > 0
                        and current_epoch >= warmup_epochs
                        and current_epoch >= self.rare_triplet_start_epoch
                    )
                    if current_epoch < warmup_epochs and not triplet_enabled:
                        batch_loss = self.autoencoders.train_on_batch(batch_x, batch_x)
                        batch_recon = batch_loss[0] if isinstance(batch_loss, (list, tuple)) else batch_loss
                        batch_triplet = 0.0
                        valid_anchors = 0.0
                    elif not triplet_enabled:
                        batch_loss = self.autoencoders.train_on_batch(
                            batch_x,
                            batch_x,
                            sample_weight=current_weights[idx],
                        )
                        batch_recon = batch_loss[0] if isinstance(batch_loss, (list, tuple)) else batch_loss
                        batch_triplet = 0.0
                        valid_anchors = 0.0
                    else:
                        batch_tensor = tf.convert_to_tensor(batch_x, dtype=tf.float32)
                        batch_weights = tf.convert_to_tensor(current_weights[idx], dtype=tf.float32)
                        batch_labels = tf.convert_to_tensor(current_triplet_labels[idx], dtype=tf.int64)
                        with tf.GradientTape() as tape:
                            recon, z = triplet_model(batch_tensor, training=True)
                            per_sample = tf.reduce_mean(tf.square(recon - batch_tensor), axis=1)
                            recon_loss = tf.reduce_mean(per_sample * batch_weights)
                            ramp_epochs = max(1, min(20, int(epochs) - int(self.rare_triplet_start_epoch)))
                            rare_loss_ramp = min(
                                1.0,
                                (current_epoch - int(self.rare_triplet_start_epoch)) / float(ramp_epochs),
                            )
                            triplet_loss, valid_tensor = self._tf_rare_triplet_loss(
                                z,
                                batch_weights,
                                batch_labels,
                            )
                            loss = recon_loss + (
                                rare_loss_ramp * float(self.rare_triplet_weight) * triplet_loss
                            )
                        grads = tape.gradient(loss, self.autoencoders.trainable_weights)
                        optimizer.apply_gradients(
                            (g, v) for g, v in zip(grads, self.autoencoders.trainable_weights) if g is not None
                        )
                        batch_loss = float(loss.numpy())
                        batch_recon = float(recon_loss.numpy())
                        batch_triplet = float(triplet_loss.numpy())
                        valid_anchors = float(valid_tensor.numpy())
                    epoch_loss += float(batch_loss) * len(idx)
                    epoch_recon += float(batch_recon) * len(idx)
                    epoch_triplet += float(batch_triplet) * len(idx)
                    epoch_valid += float(valid_anchors)

                avg_loss = epoch_loss / float(n_samples)
                avg_recon = epoch_recon / float(n_samples)
                avg_triplet = epoch_triplet / float(n_samples)
                self.pretrain_loss_history.append(avg_loss)
                self.pretrain_recon_history.append(avg_recon)
                self.pretrain_triplet_history.append(avg_triplet)
                self.pretrain_triplet_valid_anchor_history.append(epoch_valid)
                self.weight_history.append(
                    self._summarize_cell_weights(
                        current_weights,
                        epoch=current_epoch,
                        phase="warmup" if current_epoch < warmup_epochs else "weighted",
                        refreshed=weights_refreshed,
                    )
                )
                print(
                    'Epoch %d/%d, weighted AE loss: %.8f recon: %.8f triplet: %.8f'
                    % (current_epoch + 1, epochs, avg_loss, avg_recon, avg_triplet)
                )

                if self.use_earlyStop:
                    if avg_loss < (best_loss - 1e-4):
                        best_loss = avg_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= 10:
                        print('Early stopping weighted AE fine-tuning at epoch', current_epoch + 1)
                        stop_training = True
                        break
            if stop_training:
                break

    def pretrain_stacks(self, x, epochs=200,decaying_step=3):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
            decayiing_step: learning rate multiplies 0.1 every 'epochs/decaying_step' epochs 
        """
        features = x
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/decaying_step' epochs
                print ('learning rate =', pow(10, -1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')
                if self.use_earlyStop is True:
                    callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                    self.stacks[i].fit(features,features,callbacks=callbacks,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
                else:
                    self.stacks[i].fit(x=features,y=features,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        Arguments:
        x: input data, shape=(n_samples, n_dims)
        epochs: training epochs
        """
        print ('Copying layer-wise pretrained weights to deep autoencoders')
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())

        print ('Fine-tuning autoencoder end-to-end')
        if self.weighted_training:
            self._fit_autoencoder_weighted(x, epochs=epochs)
            return
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
            history = self.autoencoders.fit(x=x,y=x,callbacks=callbacks,batch_size=self.batch_size,epochs=50)
            for loss_value in history.history.get('loss', []):
                self.pretrain_loss_history.append(float(loss_value))
                self.pretrain_recon_history.append(float(loss_value))
                self.pretrain_triplet_history.append(0.0)
                self.pretrain_triplet_valid_anchor_history.append(0.0)

    def fit(self, x, epochs=300,decaying_step=3): # use stacked autoencoder pretrain and fine tuning
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)

    def fit2(self,x,epochs=300): #no stack directly tran 
        if self.weighted_training:
            self._fit_autoencoder_weighted(x, epochs=epochs)
            return
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            if self.use_earlyStop:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                history = self.autoencoders.fit(x=x,y=x,callbacks=callbacks,batch_size=self.batch_size,epochs=epochs)
            else:
                history = self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)
            for loss_value in history.history.get('loss', []):
                self.pretrain_loss_history.append(float(loss_value))
                self.pretrain_recon_history.append(float(loss_value))
                self.pretrain_triplet_history.append(0.0)
                self.pretrain_triplet_valid_anchor_history.append(0.0)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders(representation).
        
        Arguments:
        x: data
        """
        return self.encoder.predict(x)


if __name__ == "__main__":
    """
    An example for how to use SAE model on MNIST dataset. You can copy this file, and run `python3 SAE.py` in terminal
    """
    import numpy as np
    def load_mnist(sample_size=10000):
        # the data, shuffled and split between train and test sets
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        print ('MNIST samples', x.shape)
        id0=np.random.choice(x.shape[0],sample_size,replace=False)
        return x[id0], y[id0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # no use GPU
    x,y=load_mnist(10000)
    db = 'mnist'
    n_clusters = 10
    # define and train SAE model
    sae = SAE(dims=[x.shape[-1], 64,32])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights('weights_%s.h5' % db)

    # extract features
    print ('Finished training, extracting features using the trained SAE model')
    features = sae.extract_feature(x)
    print ('performing k-means clustering on the extracted features')
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)
    from sklearn.metrics import normalized_mutual_info_score as nmi
    print ('K-means clustering result on extracted features: NMI =', nmi(y, y_pred))
