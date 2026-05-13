import argparse
import xarray as xr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dataset", type=str, required=True)
    parser.add_argument("--output-dataset", type=str, required=True)
    args = parser.parse_args()

    # we use the atmos init dataset to make sure time dims correspond
    ref_ds = xr.open_dataset(args.ref_dataset, engine="zarr")

    # rename some vars to have proper coupling channels
    # THESE VALUES ARE NEVER USED! We just need to have proper coupling channels
    channel_in = ref_ds.inputs.channel_in.values
    channel_in = [c.replace("z1000", "z1000-48H") for c in channel_in]
    channel_in = [c.replace("ws10", "ws10-48H") for c in channel_in]
    channel_in = [c.replace("olr", "olr-48H") for c in channel_in]
    ref_ds = ref_ds.assign_coords(channel_in=channel_in)

    # cast channel dims as string
    ref_ds = ref_ds.assign_coords(channel_in=ref_ds.channel_in.astype(str))
    ref_ds = ref_ds.assign_coords(channel_out=ref_ds.channel_out.astype(str))
    ref_ds = ref_ds.assign_coords(channel_c=ref_ds.channel_c.astype(str))

    print(f"Writing dummy ocean init dataset to {args.output_dataset}...")
    ref_ds.to_zarr(args.output_dataset, mode="w")