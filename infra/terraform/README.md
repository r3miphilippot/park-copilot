# MongoDB Atlas as code

The database layer of Park Copilot, described with Terraform: the free **M0 cluster**, the
**application user** (least privilege: `readWrite` on `park_copilot` only) and the **network
access rule**.

These resources were first created by hand in the Atlas UI. [`imports.tf`](imports.tf) makes
Terraform **adopt** them, so the first run changes nothing: it only brings them under version
control.

## Safety nets

- `prevent_destroy` on the cluster: a change that would replace it (and lose the history) makes
  the plan fail instead.
- `ignore_changes = [password]` on the user: Atlas never returns passwords, so without it every
  plan would rotate the password and break the running app.
- No secret in the code: credentials come from environment variables; the state (which stores
  the password) and `*.tfvars` files are git-ignored.

## Usage

1. In Atlas: *Organization → Access Manager → Service Accounts → Create*, with the
   *Project Owner* role on the project. Copy the client ID and secret.
2. Export the credentials and the variables (PowerShell shown; `export` on macOS/Linux):

   ```powershell
   $env:MONGODB_ATLAS_CLIENT_ID     = "..."
   $env:MONGODB_ATLAS_CLIENT_SECRET = "..."
   $env:TF_VAR_project_id           = "..."   # Atlas: Project Settings
   $env:TF_VAR_db_password          = "..."   # the current password of the park-copilot user
   ```

3. Plan, **read the plan**, then apply:

   ```bash
   terraform init
   terraform plan    # expected: 3 to import, 0 to add, 0 to change, 0 to destroy
   terraform apply
   ```

   If the plan wants to *replace* the cluster, a variable does not match reality (usually
   `region`: `EU_WEST_3` is Paris, `EU_CENTRAL_1` Frankfurt). Fix the variable, do not force it:
   `prevent_destroy` will refuse anyway.

CI runs `terraform fmt -check` and `terraform validate` on every push (no credentials needed).

## In a team

The state is local here (single developer). In a team it would live in a remote backend
(S3 + DynamoDB lock, or HCP Terraform) so that everyone shares it and runs are locked.
