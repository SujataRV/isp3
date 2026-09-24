# CI/CD Setup — GitHub Actions Deploy

## GitHub Secrets

Go to **GitHub → your repo → Settings → Secrets and variables → Actions → New repository secret** and add all of the following.

### Backend (EC2)

| Secret | Value | Notes |
|---|---|---|
| `EC2_HOST` | e.g. `43.205.x.x` | EC2 public IP or elastic IP |
| `EC2_SSH_KEY` | Full contents of your `.pem` file | Paste entire `-----BEGIN RSA PRIVATE KEY-----` block |

### Frontend (S3)

| Secret | Value | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key | Use the `github-actions-deploy` IAM user |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key | |
| `S3_BUCKET_NAME` | e.g. `my-radar-frontend` | Bucket must already exist and be public |
| `VITE_WS_URL` | `ws://<EC2_HOST>/ws` | WebSocket endpoint for live alerts |
| `VITE_API_URL` | `http://<EC2_HOST>` | REST API base URL |
| `VITE_DEVICE_ID` | e.g. `rpi-1` | Must match the device_id sent by the RPi |

### DynamoDB (optional)

| Secret | Value | Notes |
|---|---|---|
| `USE_DYNAMODB` | `true` or `false` | Set `true` to persist FALL events to DynamoDB |

> If `USE_DYNAMODB=true`, the EC2 instance must have an **IAM role** with DynamoDB write access attached,
> OR the container must be given `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` with DynamoDB permissions.

---

## Things to Change Before First Deploy

### 1. Choose your `deployed` branch name
The workflows trigger on the branch named **`deployed`**. If you use a different branch name, update line 6 in both workflow files:

```yaml
# .github/workflows/deploy-backend.yml  (line 6)
# .github/workflows/deploy-frontend.yml (line 6)
      - deployed    # ← change this to match your branch
```

### 2. AWS Region
Both workflows hardcode `ap-south-1` (Mumbai). If your EC2/S3/DynamoDB are in a different region, update:

- `deploy-backend.yml` — the `docker run -e AWS_REGION=` line
- `deploy-frontend.yml` — the `aws-region:` line and the Done echo URL

### 3. Fall detector env vars (optional tuning)
The `docker run` command in `deploy-backend.yml` passes the default thresholds from `.env.example`.
Edit these lines in the workflow if you need different values for your environment:

```yaml
-e FALL_DESCENT_SPEED=0.5
-e FALL_MIN_PEAK_SPEED=0.7
-e FALL_MAX_HEAD_RATIO=0.70
-e FALL_MIN_HEAD_DROP=0.4
-e SENSOR_ELEV_TILT_DEG=15    # ← most likely to need changing (radar mount angle)
```

### 4. Create and configure the S3 bucket (one-time)

```bash
# 1. Create bucket
aws s3api create-bucket \
  --bucket YOUR-BUCKET-NAME \
  --region ap-south-1 \
  --create-bucket-configuration LocationConstraint=ap-south-1

# 2. Disable block-public-access
aws s3api put-public-access-block \
  --bucket YOUR-BUCKET-NAME \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# 3. Enable static website hosting
aws s3 website s3://YOUR-BUCKET-NAME/ \
  --index-document index.html \
  --error-document index.html

# 4. Add public-read bucket policy (paste into AWS Console → S3 → Permissions → Bucket policy)
```

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
  }]
}
```

### 5. IAM policy for the GitHub Actions deploy user

Attach this inline policy to your `github-actions-deploy` IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowS3Deploy",
    "Effect": "Allow",
    "Action": [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ],
    "Resource": [
      "arn:aws:s3:::YOUR-BUCKET-NAME",
      "arn:aws:s3:::YOUR-BUCKET-NAME/*"
    ]
  }]
}
```

---

## How It Works After Setup

```
git push origin deployed
      │
      ├── backend/** changed
      │     └── SCP files → EC2 → docker build → restart container
      │
      └── frontend/** changed
            └── npm ci → npm run build → aws s3 sync → live
```

**Site URL:** `http://YOUR-BUCKET-NAME.s3-website.ap-south-1.amazonaws.com`
**API health:** `http://<EC2_HOST>/health`
