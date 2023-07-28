## This queries have been used in BigQuery to extract the needed data in order to build the dashboard of Ecommerce Analysis 
## The data used comes from `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` public dataset of Google Merchandise Store.

-- Extracting Promotions Purchases and Conversions
WITH PromotionEvents AS (
  SELECT
    promotion_name AS promo_name,
    event_name,
    PARSE_DATE('%Y%m%d', event_date) AS event_date,
    user_pseudo_id,
    item.*
  FROM
    -- Replace table name.
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
    UNNEST(items) AS item
  WHERE
    -- Replace date range.
    _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name IN ('purchase', 'view_promotion')
),
AllPromotions AS (
  SELECT DISTINCT promo_name
  FROM PromotionEvents
),
AllPurchases AS (
  SELECT
    event_date,
    ap.promo_name,
    COUNT(DISTINCT user_pseudo_id) AS purchases_per_day
  FROM
    PromotionEvents pe
  RIGHT JOIN AllPromotions ap ON pe.promo_name = ap.promo_name
  WHERE pe.event_name = 'purchase'
  GROUP BY event_date, ap.promo_name
),
ViewPromotions AS (
  SELECT
    event_date,
    pe.promo_name,
    COUNT(DISTINCT user_pseudo_id) AS view_promotions_per_day
  FROM
    PromotionEvents pe
  RIGHT JOIN AllPromotions ap ON pe.promo_name = ap.promo_name
  WHERE pe.event_name = 'view_promotion'
  GROUP BY event_date, pe.promo_name
),
PurchaseRevenue AS (
  SELECT
    pe.event_date,
    pe.promo_name,
    SUM(pe.price * pe.quantity) AS purchase_revenue
  FROM PromotionEvents pe
  WHERE pe.event_name = 'purchase'
  GROUP BY pe.event_date, pe.promo_name
)
SELECT
  AllPurchases.event_date,
  CASE
    WHEN AllPurchases.promo_name IS NULL OR AllPurchases.promo_name = '(not set)' OR AllPurchases.promo_name = '' OR AllPurchases.promo_name = 'Not available in demo dataset'
    THEN 'No Promotion'
    ELSE AllPurchases.promo_name
  END AS promotion_name,
  AllPurchases.purchases_per_day,
  IFNULL(ViewPromotions.view_promotions_per_day, 0) AS view_promotions_per_day,
  ROUND(AllPurchases.purchases_per_day / (IFNULL(ViewPromotions.view_promotions_per_day, 1)) * 100, 2) AS conversion_rate,
  IFNULL(PurchaseRevenue.purchase_revenue, 0) AS purchase_revenue
FROM
  AllPurchases
LEFT JOIN ViewPromotions ON AllPurchases.event_date = ViewPromotions.event_date AND AllPurchases.promo_name = ViewPromotions.promo_name
LEFT JOIN PurchaseRevenue ON AllPurchases.event_date = PurchaseRevenue.event_date AND AllPurchases.promo_name = PurchaseRevenue.promo_name
ORDER BY event_date, AllPurchases.purchases_per_day DESC;


-- New users and Total USers of Promotions
WITH UserInfo AS (
  SELECT
    user_pseudo_id,
    MAX(IF(event_name IN ('first_visit', 'first_open'), 1, 0)) AS is_new_user
  -- Replace table name.
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  -- Replace date range.
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  GROUP BY user_pseudo_id
),
PromotionEvents AS (
  SELECT
    event_params.value.string_value AS promotion_name,
    COUNT(DISTINCT pe.user_pseudo_id) AS total_users,
    SUM(CASE WHEN ui.is_new_user = 1 THEN 1 ELSE 0 END) AS new_users
  FROM
    -- Replace table name.
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS pe
  LEFT JOIN UserInfo AS ui
  ON pe.user_pseudo_id = ui.user_pseudo_id
  CROSS JOIN UNNEST(event_params) AS event_params
  WHERE
    -- Replace date range.
    _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_params.key = 'promotion_name'
    AND event_params.value.string_value IS NOT NULL
    AND event_params.value.string_value != '(not set)'
  GROUP BY event_params.value.string_value
)
SELECT
  promotion_name,
  total_users,
  new_users
FROM
  PromotionEvents;
  

-- Purchases under promotion or not of the categories
WITH PurchaseEvents AS (
  SELECT
    event_date,
    user_pseudo_id,
    item_name,
    promotion_name,
    item_category,
    quantity
  FROM
    -- Replace table name.
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
    UNNEST(items) AS item
  WHERE
    -- Replace date range.
    _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'purchase'
)
SELECT
  event_date,
  item_category,
  SUM(CASE WHEN promotion_name IS NOT NULL AND promotion_name != '(not set)' THEN quantity ELSE 0 END) AS purchases_under_promotion,
  SUM(CASE WHEN promotion_name IS NULL OR promotion_name = '(not set)' THEN quantity ELSE 0 END) AS purchases_without_promotion
FROM
  PurchaseEvents
GROUP BY event_date, item_category
ORDER BY event_date, item_category;


-- Products bought Promo/Non-Promo
WITH Params AS (
  -- Replace with selected item_name or item_id.
  SELECT 'Google Navy Speckled Tee' AS selected_product
),
PurchaseEvents AS (
  SELECT
    user_pseudo_id,
    item.*
  FROM
    -- Replace table name.
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
    UNNEST(items) AS item
  WHERE
    -- Replace date range.
    _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'purchase'
),
PromotionStats AS (
  SELECT
    item_name,
    SUM(IF(promotion_name IS NOT NULL AND promotion_name != '(not set)', quantity, 0)) AS promo_quantity,
    SUM(IF(promotion_name IS NULL OR promotion_name = '(not set)', quantity, 0)) AS non_promo_quantity
  FROM
    PurchaseEvents
  WHERE
    item_name != (SELECT selected_product FROM Params)
  GROUP BY item_name
)
SELECT
  item_name,
  promo_quantity,
  non_promo_quantity
FROM
  PromotionStats
ORDER BY promo_quantity DESC;
