def _merge_blocks(blocks):
        """
        Helper function that merges the _blocks attribute of a ds-array into
        a single ndarray / sparse matrix.
        """
        sparse = None
        print("merge", flush=True)
        sys.stdout.write("merge")
        sys.stdout.flush()
        print(blocks[0][0].__class__.__name__ )
        print(np.array(blocks).shape)
        if np.array(blocks).shape[0]>1 and blocks[0][0].__class__.__name__ == "StorageNumpy":
            res=[]
            for block in blocks:
                value=list(block)[0]
                print(value)
                res.append(value)
            #print("res")
            print(np.array(res).shape)
            return np.concatenate(res)

        elif blocks[0][0].__class__.__name__ == "StorageNumpy":
            print("entro")
            b0 = blocks[0][0]
            #b0._is_persistent= True
            #b0._numpy_full_loaded= True
            print(b0.shape)
            print(np.array(list(b0)[0]))
            if len(b0.shape) > 2:
                return np.array(list(b0)[0])
            else:
                return np.array(list(b0))

        print("no entro")
        b0 = blocks[0][0]
        if sparse is None:
            sparse = issparse(b0)

        if sparse:
            ret = sp.bmat(blocks, format=b0.getformat(), dtype=b0.dtype)
        else:
            print("aqui")
            ret = np.block(blocks)
        print("return")
        print(ret)
        return ret

def make_persistent(self, name):
        """
        Stores data in Hecuba.

        Parameters
        ----------
        name : str
            Name of the data.

        Returns
        -------
        dsarray : ds-array
            A distributed and persistent representation of the data
            divided in blocks.
        """
        if self._sparse:
            raise Exception("Data must not be a sparse matrix.")

        x = self.collect()
        persistent_data = StorageNumpy(input_array=x, name=name)
        # self._base_array is used for much more efficient slicing.
        # It does not take up more space since it is a reference to the db.
        self._base_array = persistent_data

        blocks = []
        for block in self._blocks:
            persistent_block = StorageNumpy(input_array=block, name=name,
                                            storage_id=uuid.uuid4())
            blocks.append(persistent_block)
        self._blocks = blocks

        return self


def load_from_hecuba(name, block_size):
    """
    Loads data from Hecuba.

    Parameters
    ----------
    name : str
        Name of the data.
    block_size : (int, int)
        Block sizes in number of samples.

    Returns
    -------
    storagenumpy : StorageNumpy
        A distributed and persistent representation of the data
        divided in blocks.
    """
    persistent_data = StorageNumpy(name=name)

    bn, bm = block_size

    blocks = []
    for block in persistent_data.np_split(block_size=(bn, bm)):
        blocks.append([block])

    arr = Array(blocks=blocks, top_left_shape=block_size,
                reg_shape=block_size, shape=persistent_data.shape,
                sparse=False)
    arr._base_array = persistent_data
    return arr

def collect(self):
        """
        Collects the contents of this ds-array and returns the equivalent
        in-memory array that this ds-array represents. This method creates a
        synchronization point in the execution of the application.

        Warning: This method may fail if the ds-array does not fit in
        memory.

        Returns
        -------
        array : nd-array or spmatrix
            The actual contents of the ds-array.
        """
        self._blocks = compss_wait_on(self._blocks)
        res = self._merge_blocks(self._blocks)
        if not self._sparse:
            res = np.squeeze(res)
        return res